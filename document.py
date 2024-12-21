from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

class DocumentQA:
    # Distilbert base 
    # def __init__(self, model_name="distilbert-base-cased-distilled-squad"):
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)


    # Roberta base model
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # BERT large language model
    # def __init__(self, model_name="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"):
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def answer_question(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
        
        return answer
    
    # ----------------------------------------------------------------------------------------------

    # The rapid advancement of artificial intelligence (AI) has transformed industries across the globe. 
    # From healthcare to finance, AI systems are improving efficiency, accuracy, and decision-making processes. 
    # In healthcare, AI-powered tools assist doctors in diagnosing diseases, predicting patient outcomes, and providing personalized treatment plans. 
    # Financial institutions are leveraging AI to detect fraud, automate trading, and offer more personalized services to customers. 
    # Moreover, AI is playing a significant role in automating repetitive tasks, reducing human error, and optimizing workflows. 
    # However, the widespread adoption of AI also raises ethical concerns, such as data privacy, security, and the potential for bias in decision-making algorithms. 
    # To ensure responsible AI development, it is crucial to establish guidelines that address these concerns while fostering innovation.

    # ------------------------------------------------------------------------------------------------

    # What industries have been transformed by AI?
    # How is AI being used in healthcare?
    # What role does AI play in the financial sector?
    # How does AI improve efficiency in industries?
    # What are some ethical concerns related to AI?
    # Why is it important to establish guidelines for AI development?
    # What benefits does AI offer to financial institutions?
    # In what ways can AI help reduce human error?
    # How does AI optimize workflows?
    # What is necessary to ensure the responsible use of AI?
