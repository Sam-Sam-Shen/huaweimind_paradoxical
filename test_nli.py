from transformers import pipeline
import json

model_dir = '/workspace/huaweimind_paradoxical/models/Fengshenbang/Erlangshen-RoBERTa-330M-NLI'
pipe = pipeline('text-classification', model=model_dir, device=-1)

test_cases = [
    '我们要开放市场 [SEP] 我们要保守传统',
    '公司要快速发展 [SEP] 公司要稳健经营',
    '员工要努力奋斗 [SEP] 员工要享受生活',
    '应该进攻 [SEP] 应该防守',
]

results = []
for case in test_cases:
    result = pipe(case)
    results.append({'input': case, 'result': result})

with open('/workspace/huaweimind_paradoxical/test_nli.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("测试完成")
