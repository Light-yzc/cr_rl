import json
from pathlib import Path
import torch

root=Path('/content/crrl_training')
metadata=json.loads((root/'public_bc_full/dataset_metadata.json').read_text())
checkpoint=torch.load(root/'prepared/anchor_actor.pt',map_location='cpu',mmap=True,weights_only=False)
vocabulary=metadata['vocabulary']
print(json.dumps({'vocabulary_type':type(vocabulary).__name__,'vocabulary_len':len(vocabulary),
    'vocabulary_keys':list(vocabulary)[:5], 'model_vocab_size':checkpoint['config']['vocab_size'],
    'backend_provenance_keys':list(metadata['backend_provenance']),
    'source_content_sha':metadata['backend_provenance'].get('content_sha'),
    'model_content_sha':checkpoint.get('content_sha'),
    'source_data_version':metadata['backend_provenance'].get('data_version'),
    'model_data_version':checkpoint.get('data_version')}))
