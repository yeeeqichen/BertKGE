import torch
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer

NUM_EMBEDDINGS = 40943 + 18 + 3


class KGEncoder(torch.nn.Module):
    """
    todo: build a transformer encoder block
    """
    def __init__(self, num_head, num_layers, embedding_size, seq_length):
        self.embedding_size = embedding_size
        super(KGEncoder, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(embedding_size, num_head)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers)
        self.element_embedding = torch.nn.Embedding(
            num_embeddings=NUM_EMBEDDINGS,
            embedding_dim=embedding_size)
        self.position_embedding = torch.nn.Embedding(
            num_embeddings=seq_length,
            embedding_dim=embedding_size)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, 2),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def init_weight(self):
        init_range = 0.1
        self.element_embedding.weight.data.uniform_(-init_range, init_range)
        self.position_embedding.weight.data.uniform_(-init_range, init_range)

    def embed(self, positive_inputs, negative_inputs):
        # origin shape:
        #   (Batch_size, Sequence_length) for positive
        #   (Batch_size, Negative_sample_size, Sequence_length) for negative
        # target shape:
        #   (Sequence_length, Batch_size, Embedding_size) for positive
        #   (Sequence_length, Batch_size, Negative_sample_size, Embedding_size)  for negative
        positive_input_ids = positive_inputs['input_ids']
        positive_input_embed = self.element_embedding(positive_input_ids.transpose(1, 0))
        if negative_inputs is None:
            return {
                'candidate': positive_input_embed,
            }
        negative_input_ids = negative_inputs['input_ids']
        negative_input_embed = self.element_embedding(negative_input_ids.permute(2, 0, 1))

        return {
            'positive': positive_input_embed,
            'negative': negative_input_embed
        }

    def forward(self, input_embedding):
        # shape: (Sequence_length, Batch_size, hidden_size)
        output = self.encoder(input_embedding)
        scores = self.classifier(output[0])
        return scores

    def train_step(self, batch_input):
        input_shape = batch_input['negative']['input_ids'].shape
        batch_size = input_shape[0]
        negative_sample_size = input_shape[1]
        seq_length = input_shape[2]
        embed_outputs = self.embed(batch_input['positive'], batch_input['negative'])
        positive_embed = embed_outputs['positive']
        negative_embed = embed_outputs['negative'].view(seq_length, -1, self.embedding_size)
        positive_scores = self.forward(positive_embed)
        negative_scores = self.forward(negative_embed).view(batch_size * negative_sample_size, -1)
        positive_labels = batch_input['positive']['labels']
        negative_labels = batch_input['negative']['labels'].view(-1)
        loss, hit = self._loss(positive_scores, negative_scores, positive_labels, negative_labels)
        return loss, hit
        # print(positive_scores, negative_scores)

    def evaluate_step(self, input_candidates):
        candidate_embeds = self.embed(input_candidates, None)['candidate']
        candidate_scores = self.forward(candidate_embeds)[:, 1]
        return candidate_scores

    def _loss(self, positive_scores, negative_scores, positive_labels, negative_labels):
        positive_loss = self.loss_fn(positive_scores, positive_labels)
        negative_loss = self.loss_fn(negative_scores, negative_labels)
        pos_predict = torch.argmax(positive_scores, dim=1)
        neg_predict = torch.argmax(negative_scores, dim=1)
        hit = torch.sum(pos_predict == 1) + torch.sum(neg_predict == 0)
        return positive_loss + negative_loss, hit


if __name__ == '__main__':
    pass

