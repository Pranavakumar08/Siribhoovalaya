import numpy
from aksharamukha import transliterate
from IAST_chakra_1_1 import ch1_1
from ch1_1_BiLSTM import ch1_1_Bi

with open('./corrected/output_1_segmented_corrected.txt','r', encoding='utf-8') as f:
    RAG_op = f.readlines()

segmented_corrected_Kannada = [RAG_op[i].strip() for i in range(4,len(RAG_op))]
print(f"\nCorrect Word segmentation of Chakra 1.1 in Kannada:-\n\n{segmented_corrected_Kannada}\n\n")

segmented_corrected_IAST = []
for i in segmented_corrected_Kannada:
    word = transliterate.process('Kannada', 'IAST', i)
    segmented_corrected_IAST.append(word)
print(f"Correct Word segmentation of Chakra 1.1 in IAST:-\n\n{segmented_corrected_IAST}\n\n")

print("-" * 150)


class AccuracyScore:

    def __init__(self, predicted_words):
        self.score = 0
        self.accuracy = 0
        self.evaluate_score(predicted_words)
    
    def evaluate_score(self, predicted_words):
        for i in predicted_words:
            if i in segmented_corrected_IAST:
                self.score += 1
            else:
                if i[1:] or i[:-2] in segmented_corrected_IAST:
                    self.score += 0.67
                else:
                    if len(i) >= 3:
                        if i[2:] or i[:-3] or i[1:-2] in segmented_corrected_IAST:
                            self.score += 0.33
        self.update_acc()
    
    def update_acc(self):
        self.accuracy = self.score / len(segmented_corrected_IAST)


ch1_1_Bi_IAST = []
for i in ch1_1_Bi:
    word = transliterate.process('Kannada', 'IAST', i)
    ch1_1_Bi_IAST.append(word)
print(f"\nWord Segmentation by BiLSTM Model in IAST:-\n\n{ch1_1_Bi_IAST}\n")
bilstm = AccuracyScore(ch1_1_Bi_IAST)
print(f"BiLSTM Validation:\nScore: {bilstm.score}\nAccuracy: {bilstm.accuracy}\nAccuracy Percentage: {bilstm.accuracy*100}\n")

print("-" * 150)

print(f"\nWord Segmentation by RAG-based Model in IAST:-\n\n{ch1_1}\n")
rg = AccuracyScore(ch1_1)
print(f"RAG validation:\nScore: {rg.score}\nAccuracy: {rg.accuracy}\nAccuracy Percentage: {rg.accuracy*100}\n")