import json
import sys
sys.path.insert(0, '/workspace/huaweimind_paradoxical')

from src.data_loader import DataLoader
from src.contradiction_detect import ContradictionDetector

loader = DataLoader()
corpus = loader.load_all_documents()

documents = [
    {
        "id": doc.id,
        "title": doc.title,
        "content": doc.content,
        "year": doc.year,
    }
    for doc in corpus.documents
]

detector = ContradictionDetector()
candidate_pairs = detector.generate_candidate_pairs(documents[:100])

print(f"生成了 {len(candidate_pairs)} 个候选对")

if candidate_pairs:
    print("\n前5个候选对:")
    for i, pair in enumerate(candidate_pairs[:5]):
        print(f"\n--- 对 {i+1} ---")
        print(f"前提: {pair.premise[:80]}...")
        print(f"假设: {pair.hypothesis[:80]}...")
        
    # 测试前几个候选对
    print("\n\n测试前几个候选对:")
    test_pairs = candidate_pairs[:3]
    
    from transformers import pipeline
    model_dir = '/workspace/huaweimind_paradoxical/models/Fengshenbang/Erlangshen-RoBERTa-330M-NLI'
    pipe = pipeline('text-classification', model=model_dir, device=-1)
    
    for i, pair in enumerate(test_pairs):
        text = f"{pair.premise} [SEP] {pair.hypothesis}"
        result = pipe(text)
        print(f"\n对 {i+1}: {result}")
