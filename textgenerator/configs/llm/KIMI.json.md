```JSON
{
  id: 'Default (Custom) 1 2',
  profile: {
    extends: 'Default (Custom)',
    name: 'KIMI',
  },
  config: {
    endpoint: 'https://api.moonshot.cn/v1/chat/completions',
    custom_header: '{\n    "Content-Type": "application/json",\n    authorization: "Bearer {{api_key}}"\n}',
    custom_body: '{\n    model: "{{model}}",\n    temperature: {{temperature}},\n    top_p: {{top_p}},\n    frequency_penalty: {{frequency_penalty}},\n    presence_penalty: {{presence_penalty}},\n    max_tokens: {{max_tokens}},\n    n: {{n}},\n    stream: {{stream}},\n    stop: "{{stop}}",\n    messages: {{stringify messages}}\n}',
    model: 'moonshot-v1-8k',
    sanatization_streaming: '// catch error\nif (res.status >= 300) {\n  const err = data?.error?.message || JSON.stringify(data);\n  throw err;\n}\nlet resultTexts = [];\nconst lines = this.chunk.split("\\ndata: ");\n\nconst parsedLines = lines\n    .map((line) => line.replace(/^data: /, "").trim()) // Remove the "data: " prefix\n    .filter((line) => line !== "" && line !== "[DONE]") // Remove empty lines and "[DONE]"\n    .map((line) => {\n        try {\n            return JSON.parse(line)\n        } catch { }\n    }) // Parse the JSON string\n    .filter(Boolean);\n\nfor (const parsedLine of parsedLines) {\n    const { choices } = parsedLine;\n    const { delta } = choices[0];\n    const { content } = delta;\n    // Update the UI with the new content\n    if (content) {\n        resultTexts.push(content);\n    }\n}\nreturn resultTexts.join("");',
    sanatization_response: "// catch error\nif (res.status >= 300) {\n  const err = data?.error?.message || JSON.stringify(data);\n  throw err;\n}\n\n// get choices\nconst choices = (data.choices || data).map(c=> c.message);\n\n// the return object should be in the format of \n// { content: string }[] \n// if there's only one response, put it in the array of choices.\nreturn choices;",
    frequency_penalty: 0,
    presence_penalty: 0.5,
    top_p: 1,
    api_key: '',
    streamable: true,
  },
}
```