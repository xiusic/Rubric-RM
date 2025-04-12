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


# if __name__ == "__main__":
#     run_tests()

string_print = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.\n\nFirst, classify the task into one of two categories: <type>Reasoning</type> or <type>Chat</type>.\n- Use <type>Reasoning</type> for tasks that involve math, coding, or require domain knowledge, multi-step inference, logical deduction, or combining information to reach a conclusion.\n- Use <type>Chat</type> for tasks that involve open-ended or factual conversation, stylistic rewrites, safety questions, or general helpfulness requests without deep reasoning.\n\nIf the task is Reasoning:\n1. Solve the Client's question yourself and present your final answer within <solution>...</solution> tags.\n2. Evaluate the two Chatbot responses based on correctness, completeness, and reasoning quality, referencing your own solution.\n3. Include your evaluation inside <eval>...</eval> tags, quoting or summarizing the Chatbots using the following tags:\n - <quote_A>...</quote_A> for direct quotes from Chatbot A\n - <summary_A>...</summary_A> for paraphrases of Chatbot A\n - <quote_B>...</quote_B> for direct quotes from Chatbot B\n - <summary_B>...</summary_B> for paraphrases of Chatbot B\n4. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\nIf the task is Chat:\n1. Generate evaluation criteria (rubric) tailored to the Client's question and context, enclosed in <rubric>...</rubric> tags.\n2. Inside <rubric>, include a <justify>...</justify> section explaining why these criteria are appropriate.\n3. Compare both Chatbot responses according to the rubric.\n4. Provide your evaluation inside <eval>...</eval> tags, using <quote_A>, <summary_A>, <quote_B>, and <summary_B> as described above.\n5. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>\n\nImportant Notes:\n- Be objective and base your evaluation only on the content of the responses.\n- Do not let response order, length, or Chatbot names affect your judgment.\n- Follow the response format strictly depending on the task type.\n\nYour output must follow one of the two formats below:\n\nFor Reasoning:\n<type>Reasoning</type>\n\n<solution> your own solution for the problem </solution>\n\n<eval>\n include direct comparisons supported by <quote_A>...</quote_A> or <summary_A>...</summary_A>, and <quote_B>...</quote_B>, or <summary_B>...</summary_B>\n</eval>\n\n<answer>[[A/B]]</answer>\n\nFor Chat:\n<type>Chat</type>\n\n<rubric>\n detailed rubric items\n <justify> justification for the rubric </justify>\n</rubric>\n\n<eval>\n include direct comparisons supported by <quote_A>...</quote_A> or <summary_A>...</summary_A>, and <quote_B>...</quote_B>, or <summary_B>...</summary_B> tags\n</eval>\n\n<answer>[[A/B]]</answer>"
print(string_print)