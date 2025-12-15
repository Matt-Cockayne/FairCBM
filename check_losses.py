import json

data = json.load(open('results/test_single_1516680/fair_curriculum_cbm/history.json'))
train = data['train']
print('Epoch | Adv Loss  | Adv Lambda | Fairness Loss | Concept Loss | Binary Loss')
print('-' * 80)
for i in [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]:
    e = train[i]
    print(f"{e['epoch']:5d} | {e.get('adversarial_loss', 0):9.4f} | {e.get('adversarial_lambda', 0):10.4f} | {e.get('fairness_loss', 0):13.4f} | {e['concept_loss']:12.4f} | {e['binary_loss']:11.4f}")
