//go:build linux || darwin

package main

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

// peakRSS returns the process's peak resident set size in bytes.
func peakRSS() int64 {
	if runtime.GOOS == "linux" {
		if b, err := os.ReadFile("/proc/self/status"); err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				if strings.HasPrefix(line, "VmHWM:") {
					f := strings.Fields(line)
					kb, _ := strconv.ParseInt(f[1], 10, 64)
					return kb * 1024
				}
			}
		}
	}
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	if runtime.GOOS == "darwin" {
		return int64(ru.Maxrss)
	}
	return int64(ru.Maxrss) * 1024
}
