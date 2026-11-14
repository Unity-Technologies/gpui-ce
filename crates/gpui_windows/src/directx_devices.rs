use anyhow::{Context, Result};
use itertools::Itertools;
use std::sync::atomic::{AtomicBool, Ordering};
use windows::Win32::{
    Foundation::{E_NOTIMPL, HMODULE},
    Graphics::{
        Direct3D::{
            D3D_DRIVER_TYPE, D3D_DRIVER_TYPE_UNKNOWN, D3D_DRIVER_TYPE_WARP, D3D_FEATURE_LEVEL,
            D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
        },
        Direct3D11::{
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_DEBUG, D3D11_SDK_VERSION,
            D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
        },
        Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_DEBUG,
            DXGI_CREATE_FACTORY_FLAGS, DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED,
            DXGI_ERROR_DEVICE_RESET, DXGI_ERROR_NOT_CURRENTLY_AVAILABLE, DXGI_ERROR_NOT_FOUND,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IDXGIAdapter, IDXGIAdapter1, IDXGIDevice,
            IDXGIFactory6,
        },
    },
};
use windows::core::{HRESULT, Interface};

/// Set to `1` or `true` to draw every window with WARP, the software rasterizer Windows ships.
pub const FORCE_WARP: &str = "GPUI_FORCE_WARP";

/// Set to `1` or `true` to make renderer setup fail on a hardware adapter, as a GPU that
/// refuses the swap chain does, so the WARP retry can be checked on a working machine.
pub const SIMULATE_HARDWARE_RENDERER_FAILURE: &str = "GPUI_SIMULATE_HARDWARE_RENDERER_FAILURE";

static SOFTWARE_REQUESTED: AtomicBool = AtomicBool::new(false);
static FELL_BACK_TO_SOFTWARE: AtomicBool = AtomicBool::new(false);

/// Draw every window with WARP. Call before the platform is created, as
/// [`FORCE_WARP`] would.
pub fn request_software_renderer() {
    SOFTWARE_REQUESTED.store(true, Ordering::SeqCst);
}

/// Whether a window's renderer failed on the hardware adapter and the app moved to WARP.
pub fn fell_back_to_software_renderer() -> bool {
    FELL_BACK_TO_SOFTWARE.load(Ordering::SeqCst)
}

pub(crate) fn mark_software_fallback() {
    FELL_BACK_TO_SOFTWARE.store(true, Ordering::SeqCst);
}

fn env_flag_set(value: Option<String>) -> bool {
    value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn software_renderer_requested() -> bool {
    SOFTWARE_REQUESTED.load(Ordering::SeqCst) || env_flag_set(std::env::var(FORCE_WARP).ok())
}

/// The `DXGI_ERROR_DEVICE_REMOVED` a failing GPU returns, when
/// [`SIMULATE_HARDWARE_RENDERER_FAILURE`] is set and `devices` are on hardware.
pub(crate) fn simulated_hardware_renderer_failure(devices: &DirectXDevices) -> Result<()> {
    if devices.is_software()
        || !env_flag_set(std::env::var(SIMULATE_HARDWARE_RENDERER_FAILURE).ok())
    {
        return Ok(());
    }
    Err(
        anyhow::Error::from(windows::core::Error::from(DXGI_ERROR_DEVICE_REMOVED))
            .context(format!("{SIMULATE_HARDWARE_RENDERER_FAILURE} is set")),
    )
}

/// The HRESULTs a driver returns when its GPU cannot carry the swap chain or the first
/// resources: removed, hung or reset devices, a busy or absent adapter (`0x887A0022`), and
/// `E_NOTIMPL` from a driver missing a feature. WARP needs none of that hardware.
const SOFTWARE_RETRY_CODES: [HRESULT; 5] = [
    DXGI_ERROR_DEVICE_REMOVED,
    DXGI_ERROR_DEVICE_HUNG,
    DXGI_ERROR_DEVICE_RESET,
    DXGI_ERROR_NOT_CURRENTLY_AVAILABLE,
    E_NOTIMPL,
];

/// Whether a renderer setup failure is one WARP can avoid.
pub(crate) fn warrants_software_retry(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<windows::core::Error>())
        .any(|cause| SOFTWARE_RETRY_CODES.contains(&cause.code()))
}

/// Runs `create` on `devices`. When that fails on a hardware adapter with an HRESULT WARP can
/// avoid, builds devices with `software` and runs `create` once more on them. A failed retry
/// returns the hardware error, so the report names what broke first. The `bool` says whether
/// the result was built on the retry's devices.
pub(crate) fn retry_on_software<R>(
    devices: &DirectXDevices,
    on_software_already: bool,
    mut create: impl FnMut(&DirectXDevices) -> Result<R>,
    software: impl FnOnce() -> Result<DirectXDevices>,
) -> Result<(R, bool)> {
    let error = match create(devices) {
        Ok(result) => return Ok((result, false)),
        Err(error) => error,
    };
    if on_software_already || !warrants_software_retry(&error) {
        return Err(error);
    }
    log::warn!("Renderer setup failed on the hardware adapter, retrying on WARP: {error:#}");
    let software = match software() {
        Ok(software) => software,
        Err(software_error) => {
            log::error!("Could not create WARP devices: {software_error:#}");
            return Err(error);
        }
    };
    match create(&software) {
        Ok(result) => Ok((result, true)),
        Err(software_error) => {
            log::error!("Renderer setup failed on WARP too: {software_error:#}");
            Err(error)
        }
    }
}

pub(crate) fn try_to_recover_from_device_lost<T>(mut f: impl FnMut() -> Result<T>) -> Result<T> {
    (0..5)
        .map(|i| {
            if i > 0 {
                // Add a small delay before retrying
                std::thread::sleep(std::time::Duration::from_millis(100 + i * 10));
            }
            f()
        })
        .find_or_last(Result::is_ok)
        .unwrap()
        .context("DirectXRenderer failed to recover from lost device after multiple attempts")
}

#[derive(Clone)]
pub(crate) struct DirectXDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
}

impl DirectXDevices {
    /// Devices on the preferred hardware adapter, or on WARP once software is requested or
    /// the app has fallen back to it, so a device-lost rebuild stays on WARP.
    pub(crate) fn new() -> Result<Self> {
        if software_renderer_requested() || fell_back_to_software_renderer() {
            return Self::new_software();
        }
        Self::create(false)
    }

    /// Devices on WARP.
    pub(crate) fn new_software() -> Result<Self> {
        Self::create(true)
    }

    fn create(software_only: bool) -> Result<Self> {
        let debug_layer_available = check_debug_layer_available();
        let dxgi_factory =
            get_dxgi_factory(debug_layer_available).context("Creating DXGI factory")?;
        let devices = get_adapter(&dxgi_factory, debug_layer_available, software_only)
            .context("Getting DXGI adapter")?;
        match devices.feature_level {
            D3D_FEATURE_LEVEL_11_1 => {
                log::info!("Created device with Direct3D 11.1 feature level.")
            }
            D3D_FEATURE_LEVEL_11_0 => {
                log::info!("Created device with Direct3D 11.0 feature level.")
            }
            _ => unreachable!(),
        }

        Ok(Self {
            adapter: devices.adapter,
            dxgi_factory: devices.dxgi_factory.unwrap_or(dxgi_factory),
            device: devices.device,
            device_context: devices.device_context,
        })
    }

    pub(crate) fn is_software(&self) -> bool {
        is_software_adapter(&self.adapter)
    }
}

/// A created device. `dxgi_factory` is set only when the device came with its own factory
/// (WARP created without an adapter), which its swap chains must use.
struct CreatedDevice {
    adapter: IDXGIAdapter1,
    dxgi_factory: Option<IDXGIFactory6>,
    device: ID3D11Device,
    device_context: ID3D11DeviceContext,
    feature_level: D3D_FEATURE_LEVEL,
}

#[inline]
fn check_debug_layer_available() -> bool {
    #[cfg(debug_assertions)]
    {
        use gpui_util::ResultExt;
        use windows::Win32::Graphics::Dxgi::{DXGIGetDebugInterface1, IDXGIInfoQueue};

        unsafe { DXGIGetDebugInterface1::<IDXGIInfoQueue>(0) }
            .log_err()
            .is_some()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}

#[inline]
fn get_dxgi_factory(debug_layer_available: bool) -> Result<IDXGIFactory6> {
    let factory_flag = if debug_layer_available {
        DXGI_CREATE_FACTORY_DEBUG
    } else {
        #[cfg(debug_assertions)]
        log::warn!(
            "Failed to get DXGI debug interface. DirectX debugging features will be disabled."
        );
        DXGI_CREATE_FACTORY_FLAGS::default()
    };
    unsafe { Ok(CreateDXGIFactory2(factory_flag)?) }
}

#[inline]
fn get_adapter(
    dxgi_factory: &IDXGIFactory6,
    debug_layer_available: bool,
    software_only: bool,
) -> Result<CreatedDevice> {
    // Every adapter that was tried and why it was refused, so the final error names them.
    let mut refusals: Vec<String> = Vec::new();
    // Hardware adapters first, in DXGI's high-performance order (discrete before integrated).
    // Software adapters (WARP, the Basic Render Driver) only run if no hardware adapter works.
    let (hardware, software): (Vec<_>, Vec<_>) = list_adapters(
        |adapter_index| unsafe {
            dxgi_factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                adapter_index,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            )
        },
        &mut refusals,
    )
    .into_iter()
    .partition(|adapter| !is_software_adapter(adapter));
    let hardware = if software_only { Vec::new() } else { hardware };
    for adapter in hardware.into_iter().chain(software) {
        let gpu_name = adapter_description(&adapter);
        // Check to see whether the adapter supports Direct3D 11 and create
        // the device if it does.
        let mut context: Option<ID3D11DeviceContext> = None;
        let mut feature_level = D3D_FEATURE_LEVEL::default();
        match get_device(
            Some(&adapter),
            D3D_DRIVER_TYPE_UNKNOWN,
            Some(&mut context),
            Some(&mut feature_level),
            debug_layer_available,
        ) {
            Ok(device) => {
                log::info!("Using GPU: {gpu_name}");
                let device_context =
                    context.context("D3D11CreateDevice returned no device context")?;
                return Ok(CreatedDevice {
                    adapter,
                    dxgi_factory: None,
                    device,
                    device_context,
                    feature_level,
                });
            }
            Err(error) => {
                log::warn!("Direct3D 11 adapter {gpu_name:?} refused: {error:#}");
                refusals.push(format!("{gpu_name}: {error:#}"));
            }
        }
    }

    // DXGI lists WARP as the Basic Render Driver. When the listing failed or left it out,
    // ask Direct3D for WARP directly before giving up.
    match warp_device(debug_layer_available) {
        Ok(device) => return Ok(device),
        Err(error) => {
            log::warn!("Direct3D 11 WARP device refused: {error:#}");
            refusals.push(format!("WARP: {error:#}"));
        }
    }

    anyhow::bail!(
        "No compatible Direct3D 11 adapter found. Tried: {}",
        refusals.join("; ")
    )
}

/// Upper bound on adapter indices, so a listing that errors on every index still ends.
const MAX_ADAPTER_INDEX: u32 = 32;

/// Every adapter `next` lists until it reports `DXGI_ERROR_NOT_FOUND`. An index that fails
/// any other way is recorded in `refusals` and skipped, so a bad entry never hides the
/// adapters after it.
fn list_adapters<A>(
    mut next: impl FnMut(u32) -> windows::core::Result<A>,
    refusals: &mut Vec<String>,
) -> Vec<A> {
    let mut adapters = Vec::new();
    for adapter_index in 0..MAX_ADAPTER_INDEX {
        match next(adapter_index) {
            Ok(adapter) => adapters.push(adapter),
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(error) => {
                log::warn!("Listing DXGI adapter {adapter_index} failed: {error}");
                refusals.push(format!("adapter {adapter_index}: {error}"));
            }
        }
    }
    adapters
}

/// A WARP device created without an adapter, with the adapter and factory it came with.
fn warp_device(debug_layer_available: bool) -> Result<CreatedDevice> {
    let mut context: Option<ID3D11DeviceContext> = None;
    let mut feature_level = D3D_FEATURE_LEVEL::default();
    let device = get_device(
        None,
        D3D_DRIVER_TYPE_WARP,
        Some(&mut context),
        Some(&mut feature_level),
        debug_layer_available,
    )?;
    let device_context = context.context("D3D11CreateDevice returned no device context")?;
    let dxgi_device: IDXGIDevice = device.cast().context("Reading the WARP DXGI device")?;
    let adapter: IDXGIAdapter = unsafe { dxgi_device.GetAdapter() }?;
    let adapter: IDXGIAdapter1 = adapter.cast()?;
    let dxgi_factory: IDXGIFactory6 = unsafe { adapter.GetParent() }?;
    log::info!("Using GPU: {}", adapter_description(&adapter));
    Ok(CreatedDevice {
        adapter,
        dxgi_factory: Some(dxgi_factory),
        device,
        device_context,
        feature_level,
    })
}

fn is_software_adapter(adapter: &IDXGIAdapter1) -> bool {
    unsafe { adapter.GetDesc1() }
        .map(|desc| desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0)
        .unwrap_or(false)
}

/// The adapter's own name, such as "NVIDIA GeForce GTX 1060", when it can be
/// read.
pub(crate) fn adapter_name(adapter: &IDXGIAdapter1) -> Option<String> {
    let desc = unsafe { adapter.GetDesc1() }.ok()?;
    let name = String::from_utf16_lossy(&desc.Description)
        .trim_matches(char::from(0))
        .to_string();
    (!name.is_empty()).then_some(name)
}

fn adapter_description(adapter: &IDXGIAdapter1) -> String {
    match unsafe { adapter.GetDesc1() } {
        Ok(desc) => {
            let name = String::from_utf16_lossy(&desc.Description)
                .trim_matches(char::from(0))
                .to_string();
            format!(
                "{name} (vendor {:#06x}, device {:#06x})",
                desc.VendorId, desc.DeviceId
            )
        }
        Err(_) => "<unnamed adapter>".to_string(),
    }
}

#[inline]
fn get_device(
    adapter: Option<&IDXGIAdapter1>,
    driver_type: D3D_DRIVER_TYPE,
    context: Option<*mut Option<ID3D11DeviceContext>>,
    feature_level: Option<*mut D3D_FEATURE_LEVEL>,
    debug_layer_available: bool,
) -> Result<ID3D11Device> {
    let adapter: Option<IDXGIAdapter> = adapter.map(|adapter| adapter.cast()).transpose()?;
    let mut device: Option<ID3D11Device> = None;
    let device_flags = if debug_layer_available {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_DEBUG
    } else {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT
    };
    unsafe {
        D3D11CreateDevice(
            adapter.as_ref(),
            driver_type,
            HMODULE::default(),
            device_flags,
            // The generated shader corpus is Shader Model 5.0. Restrict device creation to
            // feature levels that can execute it instead of accepting FL10.1 and failing later
            // while constructing the renderer's first pipeline.
            Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            feature_level,
            context,
        )?;
    }
    let device = device.context("D3D11CreateDevice returned no device")?;
    // FL11.0/11.1 guarantees the compute/raw-buffer capabilities required by the SM5 DXBC
    // artifacts, so no legacy D3D10.x optional-feature probe is needed here.
    Ok(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use windows::Win32::Foundation::{E_FAIL, E_OUTOFMEMORY};

    fn hresult_error(code: HRESULT) -> anyhow::Error {
        anyhow::Error::from(windows::core::Error::from(code))
            .context("Creating DirectX resources")
            .context("Creating DirectX renderer")
    }

    #[test]
    fn the_driver_failures_seen_at_window_open_retry_on_software() {
        for code in SOFTWARE_RETRY_CODES {
            assert!(warrants_software_retry(&hresult_error(code)), "{code:?}");
        }
        // The two Sentry reports: 0x887A0022 (arm64) and 0x887A0005 (x64).
        assert!(warrants_software_retry(&hresult_error(HRESULT(
            0x887A0022_u32 as i32
        ))));
        assert!(warrants_software_retry(&hresult_error(HRESULT(
            0x887A0005_u32 as i32
        ))));
    }

    #[test]
    fn other_failures_do_not_retry() {
        assert!(!warrants_software_retry(&hresult_error(E_FAIL)));
        assert!(!warrants_software_retry(&hresult_error(E_OUTOFMEMORY)));
        assert!(!warrants_software_retry(&anyhow::anyhow!(
            "Creating DirectX renderer: no HRESULT"
        )));
    }

    #[test]
    fn a_listing_error_skips_that_index_and_keeps_listing() {
        let mut refusals = Vec::new();
        let adapters = list_adapters(
            |index| match index {
                0 => Ok("hardware"),
                1 => Err(windows::core::Error::from(
                    DXGI_ERROR_NOT_CURRENTLY_AVAILABLE,
                )),
                2 => Ok("Microsoft Basic Render Driver"),
                _ => Err(windows::core::Error::from(DXGI_ERROR_NOT_FOUND)),
            },
            &mut refusals,
        );
        assert_eq!(adapters, ["hardware", "Microsoft Basic Render Driver"]);
        assert_eq!(refusals.len(), 1);
        assert!(refusals[0].starts_with("adapter 1:"), "{refusals:?}");
    }

    #[test]
    fn a_listing_that_always_errors_still_ends() {
        let mut refusals = Vec::new();
        let adapters: Vec<()> =
            list_adapters(|_| Err(windows::core::Error::from(E_FAIL)), &mut refusals);
        assert!(adapters.is_empty());
        assert_eq!(refusals.len(), MAX_ADAPTER_INDEX as usize);
    }

    #[test]
    fn the_force_warp_value_is_one_or_true() {
        assert!(env_flag_set(Some("1".into())));
        assert!(env_flag_set(Some("TRUE".into())));
        assert!(!env_flag_set(Some("0".into())));
        assert!(!env_flag_set(Some(String::new())));
        assert!(!env_flag_set(None));
    }

    #[test]
    fn software_devices_are_warp() -> Result<()> {
        let devices = DirectXDevices::new_software()?;
        assert!(devices.is_software());
        Ok(())
    }

    #[test]
    fn a_retryable_hardware_failure_runs_once_more_on_software() -> Result<()> {
        let devices = DirectXDevices::new_software()?;
        let calls = Cell::new(0);
        let (value, fell_back) = retry_on_software(
            &devices,
            false,
            |_| {
                calls.set(calls.get() + 1);
                if calls.get() == 1 {
                    Err(hresult_error(DXGI_ERROR_DEVICE_REMOVED))
                } else {
                    Ok("drawn")
                }
            },
            DirectXDevices::new_software,
        )?;
        assert_eq!((value, fell_back, calls.get()), ("drawn", true, 2));
        Ok(())
    }

    #[test]
    fn no_retry_without_a_retryable_code_or_when_already_on_software() -> Result<()> {
        let devices = DirectXDevices::new_software()?;
        for (error_code, on_software) in [(E_FAIL, false), (DXGI_ERROR_DEVICE_REMOVED, true)] {
            let calls = Cell::new(0);
            let result: Result<((), bool)> = retry_on_software(
                &devices,
                on_software,
                |_| {
                    calls.set(calls.get() + 1);
                    Err(hresult_error(error_code))
                },
                || panic!("no software devices without a retry"),
            );
            assert!(result.is_err());
            assert_eq!(calls.get(), 1);
        }
        Ok(())
    }

    #[test]
    fn a_failed_software_retry_returns_the_hardware_error() -> Result<()> {
        let devices = DirectXDevices::new_software()?;
        let calls = Cell::new(0);
        let result: Result<((), bool)> = retry_on_software(
            &devices,
            false,
            |_| {
                calls.set(calls.get() + 1);
                if calls.get() == 1 {
                    Err(hresult_error(DXGI_ERROR_NOT_CURRENTLY_AVAILABLE))
                } else {
                    Err(hresult_error(E_FAIL))
                }
            },
            DirectXDevices::new_software,
        );
        let error = result.expect_err("both attempts failed");
        assert!(warrants_software_retry(&error), "{error:#}");
        assert_eq!(calls.get(), 2);
        Ok(())
    }
}
