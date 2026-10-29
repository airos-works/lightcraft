//! The screen's frame buffer as a buffer the GPU can copy frames into (the app's `present.rs`):
//! the raw Vulkan part, through wgpu's Vulkan backend, kept here with the crate's other `unsafe`.

/// The screen's frame buffer, imported into Vulkan: `height` rows of `row_pitch` bytes, 32-bit
/// B, G, R, A pixels.
pub struct Scanout {
    pub buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub row_pitch: u32,
}

/// Whether `device` is a Vulkan one (the frame buffer can only be imported there).
pub fn is_vulkan(device: &wgpu::Device) -> bool {
    // SAFETY: only asks whether the device is a Vulkan one; the guard is dropped at once.
    unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.is_some()
}

/// `VkImportScanoutMemoryHAIKU` (Mesa's `vk_haiku_scanout.h` in the air/OS NVK and RADV): chained
/// to `vkAllocateMemory`, the memory that comes back is the screen's frame buffer, with its shape
/// filled in.
#[repr(C)]
struct ImportScanoutMemoryHaiku {
    s_type: ash::vk::StructureType,
    p_next: *mut std::ffi::c_void,
    width: u32,
    height: u32,
    row_pitch: u32,
    size: ash::vk::DeviceSize,
}
const STRUCTURE_TYPE_IMPORT_SCANOUT_MEMORY_HAIKU: i32 = 1_000_399_000;

/// Imports the frame buffer whose rows are `bytes_per_row` long as a buffer the GPU may copy into.
/// What is copied there shows on the screen at once: only the presenter's copies into the visible
/// parts of its window, ordered against the window's direct connection, may write it.
pub fn import(device: &wgpu::Device, bytes_per_row: u32) -> Result<Scanout, String> {
    use ash::vk;
    let (buffer, memory, scanout) = {
        // SAFETY: the guard is dropped before the device is used through wgpu again.
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or("not a Vulkan device")?;
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();
        // SAFETY: the physical device is the one `hal` was made from.
        let props = unsafe { instance.get_physical_device_memory_properties(hal.raw_physical_device()) };
        let ty = props.memory_types[..props.memory_type_count as usize]
            .iter()
            .position(|t| t.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
            .ok_or("no device-local memory")? as u32;
        let mut scanout = ImportScanoutMemoryHaiku {
            s_type: vk::StructureType::from_raw(STRUCTURE_TYPE_IMPORT_SCANOUT_MEMORY_HAIKU),
            p_next: std::ptr::null_mut(),
            width: 0,
            height: 0,
            row_pitch: 0,
            size: 0,
        };
        let mut info = vk::MemoryAllocateInfo::default().allocation_size(0).memory_type_index(ty);
        info.p_next = (&raw mut scanout).cast();
        // SAFETY: `info` and the chained structure live across the call; a driver that doesn't
        // know the structure fails the zero-sized allocation.
        let memory = unsafe { raw.allocate_memory(&info, None) }.map_err(|e| format!("frame buffer import refused: {e}"))?;
        // SAFETY (all below): `memory` and `buffer` were just made on `raw` and are freed on failure.
        if scanout.row_pitch != bytes_per_row || scanout.size == 0 {
            unsafe { raw.free_memory(memory, None) };
            return Err(format!("frame buffer rows are {} bytes, the window's {bytes_per_row}", scanout.row_pitch));
        }
        // Read too, only by the presenter's PRESENT_VERIFY check.
        let info = vk::BufferCreateInfo::default().size(scanout.size).usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC);
        let buffer = match unsafe { raw.create_buffer(&info, None) } {
            Ok(b) => b,
            Err(e) => {
                unsafe { raw.free_memory(memory, None) };
                return Err(e.to_string());
            }
        };
        if let Err(e) = unsafe { raw.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                raw.destroy_buffer(buffer, None);
                raw.free_memory(memory, None);
            }
            return Err(e.to_string());
        }
        (buffer, memory, scanout)
    };
    // SAFETY: wgpu takes the buffer and its memory over and frees both once the GPU is done.
    let hal_buffer = unsafe { wgpu::hal::vulkan::Buffer::from_raw_managed(buffer, memory, 0, scanout.size) };
    let buffer = unsafe {
        device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some("present: frame buffer"),
                size: scanout.size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
    };
    Ok(Scanout { buffer, width: scanout.width, height: scanout.height, row_pitch: scanout.row_pitch })
}
