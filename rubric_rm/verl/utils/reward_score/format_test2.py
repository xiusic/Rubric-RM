import re

def check_task_format(text: str) -> bool:
    # Pattern for Chat task
    chat_pattern = re.compile(
        r'^'
        r'.*?<type>\s*Chat\s*</type>'
        r'.*?<rubric>'
        r'(?=.*?\S)'  # rubric must not be empty
        r'(?=.*?<justify>.*?\S+.*?</justify>)'
        r'(.*?)'
        r'</rubric>'
        r'.*?<eval>'
        r'(?=.*?(<quote_A>.*?\S+.*?</quote_A>|<summary_A>.*?\S+.*?</summary_A>))'
        r'(?=.*?(<quote_B>.*?\S+.*?</quote_B>|<summary_B>.*?\S+.*?</summary_B>))'
        r'(.*?)'
        r'</eval>'
        r'.*?<answer>\s*(\S.*?\S|\S)\s*</answer>'
        r'.*$',
        re.DOTALL
    )

    # Pattern for Reasoning task
    reasoning_pattern = re.compile(
        r'^'
        r'.*?<type>\s*Reasoning\s*</type>'
        r'.*?<solution>.*?\S+.*?</solution>'
        r'.*?<eval>'
        r'(?=.*?(<quote_A>.*?\S+.*?</quote_A>|<summary_A>.*?\S+.*?</summary_A>))'
        r'(?=.*?(<quote_B>.*?\S+.*?</quote_B>|<summary_B>.*?\S+.*?</summary_B>))'
        r'(.*?)'
        r'</eval>'
        r'.*?<answer>\s*(\S.*?\S|\S)\s*</answer>'
        r'.*$',
        re.DOTALL
    )

    return bool(chat_pattern.match(text)) or bool(reasoning_pattern.match(text))


def run_tests():
    test_cases = {
        "valid_chat": """
            <type>Chat</type>
            <rubric>
                Fluency, relevance
                <justify> These are appropriate for open-ended dialogue </justify>
            </rubric>
            <eval>
                <quote_A>This is a good answer.</quote_A>
                <summary_B>Response was less relevant.</summary_B>
            </eval>
            <answer>[[A]]</answer>
        """,

        "valid_reasoning": """
            <type>Reasoning</type>
            <solution>42</solution>
            <eval>
                <quote_A> The answer is 42. </quote_A>
                <summary_B> B tried but made a mistake in step 2. </summary_B>
            </eval>
            <answer>[[A]]</answer>
        """,

        "missing_type": """
            <rubric>
                Quality
                <justify>Why not?</justify>
            </rubric>
            <eval>
                <quote_A>Nice.</quote_A>
                <quote_B>Not bad.</quote_B>
            </eval>
            <answer>[[A]]</answer>
        """,

        "chat_missing_justify": """
            <type>Chat</type>
            <rubric>
                w 
                <justify>w</justify>
            </rubric>
            <eval>
                <quote_A>Okay.</quote_A>
                <quote_B>Better.</quote_B>
            </eval>
            <answer>[[B]]</answer>
        """,

        "reasoning_missing_solution": """
            <type>Reasoning</type>
            <eval>
                <quote_A>A is right</quote_A>
                <quote_B>B is wrong</quote_B>
            </eval>
            <answer>[[A]]</answer>
        """,

        "missing_references": """
            <type>Chat</type>
            <rubric>
                <justify>Fluency matters</justify>
            </rubric>
            <eval>
                No quote here.
            </eval>
            <answer>[[B]]</answer>
        """,

        "minimal_valid_chat": """
            <type>Chat</type>
            <rubric>
                helpfulness
                <justify>key for user-facing task</justify>
            </rubric>
            <eval>
                <summary_A>Good explanation.</summary_A>
                <quote_B>Not clear.</quote_B>
            </eval>
            <answer>[[A]]</answer>
        """,

        "minimal_valid_reasoning": """
            <type>Reasoning</type>
            <solution>the integral is 1</solution>
            <eval>
                <summary_A>Correct reasoning.</summary_A>
                <quote_B>Incorrect setup.</quote_B>
            </eval>
            <answer>[[A]]</answer>
        """
    }

    for name, content in test_cases.items():
        result = check_task_format(content)
        print(f"{name:<25} => {'✅ Pass' if result else '❌ Fail'}")


if __name__ == "__main__":
    run_tests()
