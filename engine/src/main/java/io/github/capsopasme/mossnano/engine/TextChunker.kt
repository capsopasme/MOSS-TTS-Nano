package io.github.capsopasme.mossnano.engine

/**
 * Port of the official voice-clone text chunking (onnx_tts_runtime.py:
 * split_voice_clone_text / split_text_by_token_budget / _prepare_text_for_sentence_chunking).
 */
object TextChunker {
    private const val SENTENCE_END = ".!?。！？；;"
    private const val CLAUSE_SPLIT = ",，、；;：:"
    private const val CLOSING = "\"'”’)]}）】》」』"
    private const val PAUSE_SHORT = 0.40
    private const val PAUSE_LONG = 0.24

    fun containsCjk(text: String): Boolean = text.any {
        it in '一'..'鿿' || it in '㐀'..'䶿' || it in '぀'..'ヿ' || it in '가'..'힯'
    }

    /** Official inter-chunk pause (by whitespace word count). */
    fun pauseSeconds(chunk: String): Double {
        val words = chunk.trim().split(Regex("\\s+")).count { it.isNotEmpty() }
        return if (words <= 4) PAUSE_SHORT else PAUSE_LONG
    }

    fun split(text: String, maxTokens: Int, countTokens: (String) -> Int): List<String> {
        val normalized = text.trim()
        if (normalized.isEmpty()) return emptyList()
        val budget = maxTokens.coerceAtLeast(1)
        val prepared = prepareForSentenceChunking(normalized)
        val sentences = splitByPunctuation(prepared, SENTENCE_END).ifEmpty { listOf(prepared.trim()) }

        val slices = ArrayList<Pair<Int, String>>()
        for (sentence in sentences) {
            val s = sentence.trim()
            if (s.isEmpty()) continue
            val n = countTokens(s)
            if (n <= budget) {
                slices += n to s
                continue
            }
            var clauses = splitByPunctuation(s, CLAUSE_SPLIT)
            if (clauses.size <= 1) clauses = listOf(s)
            for (clause in clauses) {
                val c = clause.trim()
                if (c.isEmpty()) continue
                val cn = countTokens(c)
                if (cn <= budget) {
                    slices += cn to c
                    continue
                }
                for (piece in splitByTokenBudget(c, budget, countTokens)) {
                    val pc = piece.trim()
                    if (pc.isNotEmpty()) slices += countTokens(pc) to pc
                }
            }
        }

        val chunks = ArrayList<String>()
        var current = ""
        var currentCount = 0
        for ((n, s) in slices) {
            if (current.isEmpty()) {
                current = s
                currentCount = n
                continue
            }
            if (currentCount + n > budget) {
                chunks += current.trim()
                current = s
                currentCount = n
            } else {
                current = joinParts(current, s)
                currentCount = countTokens(current)
            }
        }
        if (current.isNotEmpty()) chunks += current.trim()
        return if (chunks.size > 1) chunks else listOf(normalized)
    }

    internal fun prepareForSentenceChunking(text: String): String {
        var t = text.trim().replace("\r", " ").replace("\n", " ")
        while ("  " in t) t = t.replace("  ", " ")
        if (t.isEmpty()) return t
        if (containsCjk(t)) {
            if (t.last() !in SENTENCE_END) t += "。"
            return t
        }
        if (t[0].isLowerCase()) t = t[0].uppercaseChar() + t.substring(1)
        if (t.last().isLetterOrDigit()) t += "."
        if (t.split(' ').count { it.isNotEmpty() } < 5) t = "        $t"
        return t
    }

    internal fun splitByPunctuation(text: String, punctuation: String): List<String> {
        val out = ArrayList<String>()
        val cur = StringBuilder()
        var i = 0
        while (i < text.length) {
            val ch = text[i]
            cur.append(ch)
            if (ch in punctuation) {
                var j = i + 1
                while (j < text.length && text[j] in CLOSING) {
                    cur.append(text[j])
                    j++
                }
                val s = cur.toString().trim()
                if (s.isNotEmpty()) out += s
                cur.setLength(0)
                while (j < text.length && text[j].isWhitespace()) j++
                i = j
                continue
            }
            i++
        }
        val tail = cur.toString().trim()
        if (tail.isNotEmpty()) out += tail
        return out
    }

    internal fun splitByTokenBudget(text: String, maxTokens: Int, countTokens: (String) -> Int): List<String> {
        var remaining = text.trim()
        val pieces = ArrayList<String>()
        val preferred = CLAUSE_SPLIT + SENTENCE_END + " "
        while (remaining.isNotEmpty()) {
            if (countTokens(remaining) <= maxTokens) {
                pieces += remaining
                break
            }
            var low = 1
            var high = remaining.length
            var best = 1
            while (low <= high) {
                val mid = (low + high) / 2
                val candidate = remaining.substring(0, mid).trim()
                if (candidate.isEmpty()) {
                    low = mid + 1
                    continue
                }
                if (countTokens(candidate) <= maxTokens) {
                    best = mid
                    low = mid + 1
                } else {
                    high = mid - 1
                }
            }
            var cut = best
            val prefix = remaining.substring(0, best)
            val scanMin = maxOf(-1, prefix.length - 25)
            var idx = prefix.length - 1
            while (idx > scanMin) {
                if (prefix[idx] in preferred) {
                    cut = idx + 1
                    break
                }
                idx--
            }
            var piece = remaining.substring(0, cut).trim()
            if (piece.isEmpty()) {
                piece = remaining.substring(0, best).trim()
                cut = best
            }
            pieces += piece
            remaining = remaining.substring(cut).trim()
        }
        return pieces
    }

    private fun joinParts(left: String, right: String): String = when {
        left.isEmpty() -> right
        right.isEmpty() -> left
        containsCjk(left) || containsCjk(right) -> left + right
        else -> "$left $right"
    }
}
