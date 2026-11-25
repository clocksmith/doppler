//go:build windows

package main

import (
	"syscall"
	"unsafe"
)

type processMemoryCounters struct {
	Cb                         uint32
	PageFaultCount             uint32
	PeakWorkingSetSize         uintptr
	WorkingSetSize             uintptr
	QuotaPeakPagedPoolUsage    uintptr
	QuotaPagedPoolUsage        uintptr
	QuotaPeakNonPagedPoolUsage uintptr
	QuotaNonPagedPoolUsage     uintptr
	PagefileUsage              uintptr
	PeakPagefileUsage          uintptr
}

// peakRSS returns the process's peak working set in bytes.
func peakRSS() int64 {
	psapi := syscall.NewLazyDLL("psapi.dll")
	proc := psapi.NewProc("GetProcessMemoryInfo")
	h, _ := syscall.GetCurrentProcess()
	var c processMemoryCounters
	c.Cb = uint32(unsafe.Sizeof(c))
	r, _, _ := proc.Call(uintptr(h), uintptr(unsafe.Pointer(&c)), uintptr(c.Cb))
	if r == 0 {
		return -1
	}
	return int64(c.PeakWorkingSetSize)
}
