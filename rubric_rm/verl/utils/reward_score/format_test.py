import re

def get_pattern():
    """
    Returns the compiled regex pattern enforcing:
    1) <rubric> with at least one non-whitespace char,
       and a non-empty <justify> inside it.
    2) <eval> referencing Chatbot A and B at least once each with non-empty content.
    3) <answer> with non-empty text.
    """
    pattern = re.compile(
        r'^'
        # --- 1) <rubric> ... </rubric> ---
        r'.*?<rubric>'
        # Lookahead #1: ensure <rubric> block has at least one non-whitespace character
        r'(?=.*?\S)'
        # Lookahead #2: ensure there's a <justify> with non-empty text inside <rubric>
        r'(?=.*?<justify>\s*\S.*?\S\s*</justify>)'
        # Actually match/capture the <rubric> content (non-greedy)
        r'(.*?)'
        r'</rubric>'

        # --- 2) <eval> ... </eval> ---
        r'.*?<eval>'
        # Lookahead #1: must have non-empty reference to Chatbot A
        r'(?=.*?(?:<quote_A>\s*\S.*?\S\s*</quote_A>|<summary_A>\s*\S.*?\S\s*</summary_A>))'
        # Lookahead #2: must have non-empty reference to Chatbot B
        r'(?=.*?(?:<quote_B>\s*\S.*?\S\s*</quote_B>|<summary_B>\s*\S.*?\S\s*</summary_B>))'
        # Actually match/capture the <eval> content (non-greedy)
        r'(.*?)'
        r'</eval>'

        # --- 3) <answer> ... </answer> ---
        # Must have non-empty content inside <answer> tags
        r'.*?<answer>\s*(\S.*?\S|\S)\s*</answer>'
        r'.*$',
        re.DOTALL
    )
    return pattern


def test_pattern():
    """
    Tests the regex pattern against various valid and invalid cases,
    printing pass/fail results for each one.
    """
    pattern = get_pattern()

    test_cases = [
        # ========== VALID Cases ==========

        # 1. Minimal valid structure
        (
            """
<rubric>
  Rubric item
  <justify>We justify the rubric</justify>
</rubric>
<eval>
  <quote_A>Some text from A</quote_A>
  <quote_B>Some text from B</quote_B>
</eval>
<answer>[[A]]</answer>
""",
            True,
            "Minimal valid structure with non-empty rubric, justify, A & B references, and answer"
        ),

        # 2. Uses summary for A, quote for B; multiline justification
        (
            """
<rubric>
  Rubric item 1
  Rubric item 2
  <justify>
    This is a multiline
    justification for the rubric.
  </justify>
</rubric>
<eval>
  Here is an evaluation.
  <summary_A>A was concise but correct.</summary_A>
  Some extra text ...
  <quote_B>Exact words from B here</quote_B>
</eval>
<answer>[[B]]</answer>
""",
            True,
            "Valid: summary_A and quote_B used, non-empty justification"
        ),

        # 3. Multiple references, demonstrates everything is present
        (
            """
[response] <rubic>
  <item>Relevance to the Question</item>
  <item>Correct Mathematical Approach</item>
  <item>Understanding of Missing Parameters Issue</item>
  <justify>
    The first criterion evaluates how directly the response addresses the client's question. The second assesses whether the response uses appropriate mathematical principles to address the problem or not. The third examines if the chatbot understands the core issue: the inadequacy of given parameters to solve for unknowns.
  </justify>
</rubic>
<eval>
  Neither chatbot provides a relevant response to the client's question or demonstrates understanding of the geometric problem or parameters given. Both vent into unrelated, albeit sound mathematical proofs.
  <quote_A>Assume $$\cos\frac1x=\cos\frac1{x+T}.$$ Then we must have $$\frac1x=\pm\frac1{x+T}+2k\pi,$$ $$k=\frac{\dfrac1x\mp\dfrac1{x+T}}{2\pi}.$$ But the RHS cannot be an integer for all $x$, a contradiction.</quote_A>
  - Chatbot A offers a proof involving the cosine function and its periodicity, which does not address the geometric problem at hand, nor does it provide an answer to the question.
  <quote_B>For $x>\frac2\pi$ we have $0 < \frac1x<\frac\pi2$ and hence  $f(x)=\cos\frac 1x>0$, whereas $f(\frac1{\pi})=\cos\pi=-1$. If $T>0$ is a period of $f$, pick $n\in\Bbb N$ with $nT>\frac2\pi$ and arrive at the contradiction $-1=f(\frac1\pi)=f(\frac1\pi+nT)>0$.</quote_B>
  - Similarly, Chatbot B presents a proof demonstrating that $\cos \frac{1}{x}$ cannot have a non-zero period greater than $\frac{2}{\pi}$, which is not related to the initial problem of calculating the angle given the arc length and sagitta length.
  Both chatbots fail in implementing a logical response to the client's issue, showing that neither recognizes nor responds appropriately to the issue of insufficient parameters given to solve for an angle.
</eval>
<answer>[[B]]</answer>
""",
            True,
            "Valid: multiple references, clear rubric, justification, answer"
        ),

        # ========== INVALID Cases ==========

        # 4. Missing <justify> entirely in <rubric>
        (
            """
<rubric>
  Rubric item 1
  Rubric item 2
</rubric>
<eval>
  <quote_A>Text A</quote_A>
  <quote_B>Text B</quote_B>
</eval>
<answer>[[A]]</answer>
""",
            False,
            "Invalid: no <justify> tag inside <rubric>"
        ),

        # 5. Empty <justify> (only whitespace)
        (
            """
<rubric>
  Some rubric text
  <justify>   </justify>
</rubric>
<eval>
  <quote_A>A reference</quote_A>
  <quote_B>B reference</quote_B>
</eval>
<answer>[[B]]</answer>
""",
            False,
            "Invalid: <justify> is empty/whitespace only"
        ),

        # 6. Rubric is minimal, but <justify> is non-empty -> Valid
        (
            """
[Response] <rubric> www
  <justify>Non-empty justification</justify>
</rubric>
<eval>
  <quote_A>A text</quote_A>
  <quote_B>B text</quote_B>
</eval>
<answer>[[A]]</answer>
""",
            True,
            "Valid: minimal rubric but with non-empty <justify>, valid references, and answer"
        ),

        # 7. No references to Chatbot A
        (
            """
<rubric>
  1) Some rubric
  <justify>This is a justification.</justify>
</rubric>
<eval>
  <quote_B>Text from B</quote_B>
</eval>
<answer>[[B]]</answer>
""",
            False,
            "Invalid: no reference to A in <eval>"
        ),

        # 8. No references to Chatbot B
        (
            """
<rubric>
  Some rubric item
  <justify>We justify the rubric here</justify>
</rubric>
<eval>
  <quote_A>Text from A</quote_A>
</eval>
<answer>[[A]]</answer>
""",
            False,
            "Invalid: no reference to B in <eval>"
        ),

        # 9. Empty <quote_A> and valid <quote_B>
        (
            """
<rubric>
  Some rubric
  <justify>Justification text</justify>
</rubric>
<eval>
  <quote_A>   </quote_A>
  <quote_B>Valid text from B</quote_B>
</eval>
<answer>[[A]]</answer>
""",
            False,
            "Invalid: <quote_A> is only whitespace"
        ),

        # 10. Missing <answer> entirely
        (
            """
<rubric>
  Some rubric content
  <justify>Non-empty justification</justify>
</rubric>
<eval>
  <quote_A>Some content from A</quote_A>
  <quote_B>Some content from B</quote_B>
</eval>
""",
            False,
            "Invalid: No <answer> block"
        ),
    ]

    # Execution: match each test case against the pattern
    for i, (test_input, expected, description) in enumerate(test_cases, start=1):
        match_result = bool(pattern.match(test_input.strip()))
        result = "PASS" if match_result == expected else "FAIL"
        print(f"Test {i} - {description}\nExpected: {expected}, Got: {match_result} => {result}\n")


# Uncomment this line to run tests directly in a script:
test_pattern()
