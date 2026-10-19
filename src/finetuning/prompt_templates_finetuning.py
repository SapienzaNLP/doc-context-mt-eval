
###########################################################################################
SYSTEM_MESSAGE = """You are an expert annotator tasked with evaluating the quality of machine translation."""


###########################################################################################
BASE_REF_LESS_TEMPLATE = """{source_language} Source Segment:
```{source_para}```

{target_language} Translation Segment:
```{target_para}```

You are given a source segment and its machine translation segment, enclosed in triple backticks. 
Your task is to identify and classify the errors in the translation segment by comparing it to the source segment.
The categories and subcategories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, or other.

Each error is classified by severity:  
- critical: completely changes the meaning or inhibits comprehension of the text.
- major: notably changes the meaning or disrupts readability.
- minor: slightly changes the meaning or affects readability.

For each error, provide:  
1. The minimal error span: the smallest contiguous string in the translation that contains the error, without including surrounding correct text.
2. The extended span with context: exactly 4 words before and 4 words after the minimal error span, taken directly from the translation as they appear. If there are fewer than 4 words before or after, include as many as available.
3. The error type (error category and subcategory).
4. The error severity (critical, major or minor).

Use this exact format in your final answer:

{{
  "errors": [
    {{
      "minimal_span": "<minimal error span of text that contains the error>",
      "extended_span": "<extended span of text that contains the error>",
      "type": "<error category>/<error subcategory>",
      "severity": "<critical|major|minor>"
    }}
  ]
}}

If no errors are found, return {{"errors": []}}.
"""

###########################################################################################
BASE_REF_BASED_TEMPLATE = """{source_language} Source Segment:
```{source_para}```

{target_language} Reference Segment:
```{reference_para}```

{target_language} Translation Segment:
```{target_para}```

You are given a source segment, its human reference segment, and its machine translation segment, enclosed in triple backticks. 
Your task is to identify and classify the errors in the translation segment by comparing it to the source segment and the reference segment.
The categories and subcategories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, or other.

Each error is classified by severity:  
- critical: completely changes the meaning or inhibits comprehension of the text.
- major: notably changes the meaning or disrupts readability.
- minor: slightly changes the meaning or affects readability.

For each error, provide:  
1. The minimal error span: the smallest contiguous string in the translation that contains the error, without including surrounding correct text.
2. The extended span with context: exactly 4 words before and 4 words after the minimal error span, taken directly from the translation as they appear. If there are fewer than 4 words before or after, include as many as available.
3. The error type (error category and subcategory).
4. The error severity (critical, major or minor).

Use this exact format in your final answer:

{{
  "errors": [
    {{
      "minimal_span": "<minimal error span of text that contains the error>",
      "extended_span": "<extended span of text that contains the error>",
      "type": "<error category>/<error subcategory>",
      "severity": "<critical|major|minor>"
    }}
  ]
}}

If no errors are found, return {{"errors": []}}.
"""


###########################################################################################
DOC_REF_LESS_TEMPLATE = """{source_language} Full Source Document:
```{source_doc}```

{target_language} Full Translation Document:
```{target_doc}```

{source_language} Source Segment:
```{source_para}```

{target_language} Translation Segment:
```{target_para}```

You are given a source segment, its machine translation segment, and their corresponding full documents, enclosed in triple backticks. 
Your task is to identify and classify the errors in the translation segment by comparing it to the source segment, while using the full documents to resolve contextual ambiguities.
The categories and subcategories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, or other.

Each error is classified by severity:  
- critical: completely changes the meaning or inhibits comprehension of the text.
- major: notably changes the meaning or disrupts readability.
- minor: slightly changes the meaning or affects readability.

For each error, provide:  
1. The minimal error span: the smallest contiguous string in the translation that contains the error, without including surrounding correct text.
2. The extended error span: the minimal error span extended by a few words before and after, taken directly from the translation as they appear.
3. The error type (error category and subcategory).
4. The error severity (critical, major or minor).

Use this exact format in your final answer:

{{
  "errors": [
    {{
      "minimal_span": "<minimal span of text that contains the error>",
      "extended_span": "<extended span of text that contains the error>",
      "type": "<error category>/<error subcategory>",
      "severity": "<critical|major|minor>"
    }}
  ]
}}

If no errors are found, return {{"errors": []}}.
"""


###########################################################################################
DOC_REF_BASED_TEMPLATE = """{source_language} Full Source Document:
```{source_doc}```

{target_language} Full Translation Document:
```{target_doc}```

{source_language} Source Segment:
```{source_para}```

{target_language} Reference Segment:
```{reference_para}```

{target_language} Translation Segment:
```{target_para}```

You are given a source segment, its human reference segment, its machine translation segment, and the corresponding full source and translation documents, enclosed in triple backticks.
Your task is to identify and classify the errors in the translation segment by comparing it to the source segment, while using the full documents to resolve contextual ambiguities.
The categories and subcategories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, or other.

Each error is classified by severity:  
- critical: completely changes the meaning or inhibits comprehension of the text.
- major: notably changes the meaning or disrupts readability.
- minor: slightly changes the meaning or affects readability.

For each error, provide:  
1. The minimal error span: the smallest contiguous string in the translation that contains the error, without including surrounding correct text.
2. The extended span with context: exactly 4 words before and 4 words after the minimal error span, taken directly from the translation as they appear. If there are fewer than 4 words before or after, include as many as available.
3. The error type (error category and subcategory).
4. The error severity (critical, major or minor).

Use this exact format in your final answer:

{{
  "errors": [
    {{
      "minimal_span": "<minimal error span of text that contains the error>",
      "extended_span": "<extended span of text that contains the error>",
      "type": "<error category>/<error subcategory>",
      "severity": "<critical|major|minor>"
    }}
  ]
}}

If no errors are found, return {{"errors": []}}.
"""


###########################################################################################