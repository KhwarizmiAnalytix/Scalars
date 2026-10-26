#pragma once

// Page-aligned shared storage for Metal's newBufferWithBytesNoCopy. Tests and
// benchmarks own these allocations; the library does not.

#include <mach/mach.h>
#include <unistd.h>

#include <cstddef>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>

namespace scalars::metal_storage
{

inline std::size_t page_multiple(std::size_t bytes)
{
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
    {
        throw std::runtime_error("Metal test storage: page size unavailable");
    }
    const auto page_size = static_cast<std::size_t>(page);
    if (bytes == 0)
    {
        return page_size;
    }
    const std::size_t remainder = bytes % page_size;
    if (remainder == 0)
    {
        return bytes;
    }
    return bytes + (page_size - remainder);
}

inline std::mutex& registry_mutex()
{
    static std::mutex mutex;
    return mutex;
}

inline std::unordered_map<void*, std::size_t>& registry()
{
    static std::unordered_map<void*, std::size_t> sizes;
    return sizes;
}

inline void* allocate(std::size_t bytes)
{
    const vm_size_t     size    = static_cast<vm_size_t>(page_multiple(bytes));
    vm_address_t        address = 0;
    const kern_return_t result  = vm_allocate(mach_task_self(), &address, size, VM_FLAGS_ANYWHERE);
    if (result != KERN_SUCCESS)
    {
        throw std::bad_alloc();
    }
    void* ptr = reinterpret_cast<void*>(address);
    try
    {
        const std::lock_guard<std::mutex> lock(registry_mutex());
        registry().emplace(ptr, static_cast<std::size_t>(size));
    }
    catch (...)
    {
        vm_deallocate(mach_task_self(), address, size);
        throw;
    }
    return ptr;
}

inline void deallocate(void* ptr) noexcept
{
    if (ptr == nullptr)
    {
        return;
    }
    std::size_t size = 0;
    {
        const std::lock_guard<std::mutex> lock(registry_mutex());
        const auto                        it = registry().find(ptr);
        if (it == registry().end())
        {
            return;
        }
        size = it->second;
        registry().erase(it);
    }
    vm_deallocate(
        mach_task_self(), reinterpret_cast<vm_address_t>(ptr), static_cast<vm_size_t>(size));
}

}  // namespace scalars::metal_storage
