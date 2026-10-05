package io.github.capsopasme.mossnano.engine

import java.io.File

/** Loads libmossnano_jni (SentencePiece + affinity helpers) once per process. */
internal object NativeLib {
    @Volatile private var loaded = false

    @Synchronized
    fun ensureLoaded() {
        if (loaded) return
        val override = SpmTokenizer.libraryPathOverride
        if (override != null) System.load(override) else System.loadLibrary("mossnano_jni")
        loaded = true
    }
}

/**
 * Keeps the inference threads on the performance cores.
 *
 * On a big.LITTLE SoC such as the Snapdragon 8 Gen 3 (1×X4 + 5×A720 + 2×A520) a single
 * ONNX Runtime worker landing on an A520 stalls every parallel section, because all
 * workers wait for the slowest one. We therefore exclude the slowest cluster.
 *
 * Linux threads inherit the affinity of the thread that creates them, so pinning the
 * thread that creates the ORT environment / sessions pins ORT's pool threads too.
 * Everything here is best effort: on failure nothing changes.
 */
object CpuAffinity {
    /** Mask of all cores except the slowest cluster; 0 when unknown or homogeneous. */
    val performanceMask: Long by lazy { runCatching { computePerformanceMask(File("/sys/devices/system/cpu")) }.getOrDefault(0L) }

    /** Number of cores in [performanceMask] (0 when unknown). */
    val performanceCores: Int get() = java.lang.Long.bitCount(performanceMask)

    /** Current thread's mask, or 0 if it can't be read. */
    fun currentThreadMask(): Long = runCatching {
        NativeLib.ensureLoaded()
        nativeGetCurrentThreadMask()
    }.getOrDefault(0L)

    /** Sets the current thread's mask. Returns false (and changes nothing) on failure. */
    fun setCurrentThreadMask(mask: Long): Boolean {
        if (mask == 0L) return false
        return runCatching {
            NativeLib.ensureLoaded()
            nativeSetCurrentThreadMask(mask) == 0
        }.getOrDefault(false)
    }

    /**
     * Runs [block] with the current thread pinned to the performance cores and restores the
     * previous mask afterwards.
     */
    inline fun <T> onPerformanceCores(enabled: Boolean = true, block: () -> T): T {
        val target = if (enabled) performanceMask else 0L
        val previous = if (target != 0L) currentThreadMask() else 0L
        val pinned = previous != 0L && setCurrentThreadMask(target)
        try {
            return block()
        } finally {
            if (pinned) setCurrentThreadMask(previous)
        }
    }

    internal fun computePerformanceMask(cpuRoot: File): Long {
        val possible = File(cpuRoot, "possible").takeIf { it.isFile }?.readText()?.trim() ?: return 0L
        val cpus = parseCpuList(possible).filter { it in 0..63 }
        if (cpus.size < 3) return 0L
        // Prefer the scheduler's capacity (ARM), fall back to the max frequency.
        fun readLong(f: File): Long? = runCatching { f.readText().trim().toLong() }.getOrNull()
        val perf = cpus.associateWith { c ->
            val dir = File(cpuRoot, "cpu$c")
            readLong(File(dir, "cpu_capacity")) ?: readLong(File(dir, "cpufreq/cpuinfo_max_freq"))
        }
        if (perf.values.any { it == null }) return 0L
        val values = perf.values.filterNotNull()
        val slowest = values.minOrNull() ?: return 0L
        if (values.all { it == slowest }) return 0L // homogeneous: nothing to exclude
        val fast = perf.filterValues { it != null && it > slowest }.keys
        if (fast.size < 2) return 0L
        return fast.fold(0L) { m, c -> m or (1L shl c) }
    }

    internal fun parseCpuList(s: String): List<Int> = s.split(',').flatMap { part ->
        val p = part.trim()
        if (p.isEmpty()) emptyList()
        else if ('-' in p) {
            val (a, b) = p.split('-', limit = 2).map { it.trim().toInt() }
            (a..b).toList()
        } else listOf(p.toInt())
    }

    @JvmStatic private external fun nativeSetCurrentThreadMask(mask: Long): Int
    @JvmStatic private external fun nativeGetCurrentThreadMask(): Long
}
