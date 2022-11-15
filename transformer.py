
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss
from torch.optim import Adam

class PositionalEncodingLayer(Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        # inspired by the pytorch tutorial 
        # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """
        # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        T = X.shape[1]
        D = self.embedding_dim
        pos = torch.arange(T)
        pos = pos.unsqueeze(1)
        P = torch.zeros(1,T,D)
        odd_even = torch.exp(torch.arange(0, D, 2) * (-math.log(10000) / D))
        P[0,:,0::2] = torch.sin(pos * odd_even)
        P[0,:,1::2] = torch.cos(pos * odd_even)
        X = X + P
        return X

        


class SelfAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.linear_Q = Linear(in_dim, out_dim)
        self.linear_K = Linear(in_dim, out_dim)
        self.linear_V = Linear(in_dim, out_dim)

        self.softmax = Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query_X, key_X and value_X have shape (batch_size, sequence_length, in_dim). The sequence length
        may be different for query_X and key_X but must be the same for key_X and value_X.

        This function should return two things:
            - The output of the self-attention, which will have shape (batch_size, sequence_length, out_dim)
            - The attention weights, which will have shape (batch_size, query_sequence_length, key_sequence_length)

        If a mask is passed as input, you should mask the input to the softmax, using `float(-1e32)` instead of -infinity.
        The mask will be a tensor with 1's and 0's, where 0's represent entries that should be masked (set to -1e32).

        Hint: The following functions may be useful
            - torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html)
            - torch.Tensor.masked_fill (https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
        """
        query_X = self.linear_Q(query_X)
        key_X = self.linear_K(key_X)
        value_X = self.linear_V(value_X)

        key_T = torch.transpose(key_X, 1, 2)
        QK_T = torch.bmm(query_X,key_T)
        sqrt_dk = math.sqrt(self.out_dim)

        QK_T = QK_T/sqrt_dk
        if mask == None:
            attention_weights = self.softmax(QK_T)
            output = torch.bmm(attention_weights,value_X)
            return (output,attention_weights)
        else:
            QK_T[mask == 0] = float(-1e32)
            attention_weights = self.softmax(QK_T)
            output = torch.bmm(attention_weights,value_X)
            return (output,attention_weights)

  


class MultiHeadedAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention_heads = ModuleList([SelfAttentionLayer(in_dim, out_dim) for _ in range(n_heads)])

        self.linear = Linear(n_heads * out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention
        and the attention weights of each attention head.

        The attention_weights matrix has dimensions (batch_size, heads, query_sequence_length, key_sequence_length)
        """

        outputs, attention_weights = [], []

        for attention_head in self.attention_heads:
            out, attention = attention_head(query_X, key_X, value_X, mask)
            outputs.append(out)
            attention_weights.append(attention)

        outputs = torch.cat(outputs, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return self.linear(outputs), attention_weights
        
class EncoderBlock(Module):

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def forward(self, X, mask=None):
        """
        Implementation of an encoder block. Both the input and output
        have shape (batch_size, source_sequence_length, embedding_dim).

        The mask is passed to the multi-headed self-attention layer,
        and is usually used for the padding in the encoder.
        """  
        att_out, _ = self.attention(X, X, X, mask)

        residual = X + self.dropout1(att_out)

        X = self.norm1(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)

        residual = X + self.dropout2(temp)

        return self.norm2(residual)

class Encoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([EncoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])
        self.vocab_size = vocab_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder. The input has dimensions (batch_size, sequence_length)
        and the output has dimensions (batch_size, sequence_length, embedding_dim).

        The encoder returns its output and the location of the padding, which will be
        used by the decoder.
        """

        padding_locations = torch.where(X == self.vocab_size, torch.zeros_like(X, dtype=torch.float64),
                                        torch.ones_like(X, dtype=torch.float64))
        padding_mask = torch.einsum("bi,bj->bij", (padding_locations, padding_locations))

        X = self.embedding_layer(X)
        X = self.position_encoding(X)
        for block in self.blocks:
            X = block(X, padding_mask)
        return X, padding_locations

class DecoderBlock(Module):

    def __init__(self, embedding_dim, n_heads) -> None:
        super().__init__()

        self.attention1 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)
        self.attention2 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)

    def forward(self, encoded_source: torch.Tensor, target: torch.Tensor,
                mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of a decoder block. encoded_source has dimensions (batch_size, source_sequence_length, embedding_dim)
        and target has dimensions (batch_size, target_sequence_length, embedding_dim).

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, target_sequence_length, embedding_dim) and
        the attention matrices for each of the heads of the second multi-headed self-attention layer
        (the one where the source and target are "mixed").
        """  
        att_out, _ = self.attention1(target, target, target, mask1)
        residual = target + self.dropout1(att_out)
        
        X = self.norm1(residual)

        att_out, att_weights = self.attention2(X, encoded_source, encoded_source, mask2)

        residual = X + self.dropout2(att_out)
        X = self.norm2(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)
        residual = X + self.dropout3(temp)

        return self.norm3(residual), att_weights

class Decoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()
        
        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([DecoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

        self.linear = Linear(embedding_dim, vocab_size + 1)
        self.softmax = Softmax(-1)

        self.vocab_size = vocab_size

    def _lookahead_mask(self, seq_length: int) -> torch.Tensor:
        """
        Compute the mask to prevent the decoder from looking at future target values.
        The mask you return should be a tensor of shape (sequence_length, sequence_length)
        with only 1's and 0's, where a 0 represent an entry that will be masked in the
        multi-headed attention layer.

        Hint: The function torch.tril (https://pytorch.org/docs/stable/generated/torch.tril.html)
        may be useful.
        """
        mask = torch.ones(seq_length,seq_length)
        mask = torch.tril(mask)
        return mask

    def forward(self, encoded_source: torch.Tensor, source_padding: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Transformer decoder. encoded_source has dimensions (batch_size, source_sequence_length, embedding),
        source_padding has dimensions (batch_size, source_seuqence_length) and target has dimensions
        (batch_size, target_sequence_length).

        Returns its output of shape (batch_size, target_sequence_length, target_vocab_size) and
        the attention weights from the first decoder block, of shape
        (batch_size, n_heads, source_sequence_length, target_sequence_length)

        Note that the output is not normalized (i.e. we don't use the softmax function).
        """
        
        # Lookahead mask
        seq_length = target.shape[1]
        mask = self._lookahead_mask(seq_length)

        # Padding masks
        target_padding = torch.where(target == self.vocab_size, torch.zeros_like(target, dtype=torch.float64), 
                                     torch.ones_like(target, dtype=torch.float64))
        target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, target_padding))
        mask1 = torch.multiply(mask, target_padding_mask)

        source_target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, source_padding))

        target = self.embedding_layer(target)
        target = self.position_encoding(target)

        att_weights = None
        for block in self.blocks:
            target, att = block(encoded_source, target, mask1, source_target_padding_mask)
            if att_weights is None:
                att_weights = att

        y = self.linear(target)
        return y, att_weights


class Transformer(Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int, n_encoder_blocks: int,
                 n_decoder_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.encoder = Encoder(source_vocab_size, embedding_dim, n_encoder_blocks, n_heads)
        self.decoder = Decoder(target_vocab_size, embedding_dim, n_decoder_blocks, n_heads)
        self.softmax = Softmax(-1)


    def forward(self, source, target):
        encoded_source, source_padding = self.encoder(source)
        return self.decoder(encoded_source, source_padding, target)

    def predict(self, source: List[int], beam_size=1, max_length=12) -> List[int]:
        """
        Given a sentence in the source language, you should output a sentence in the target
        language of length at most `max_length` that you generate using a beam search with
        the given `beam_size`.

        Note that the start of sentence token is 0 and the end of sentence token is 1.

        Return the final top beam (decided using average log-likelihood) and its average
        log-likelihood.

        Hint: The follow functions may be useful:
            - torch.topk (https://pytorch.org/docs/stable/generated/torch.topk.html)
            - torch.softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
        """

        self.eval() # Set the PyTorch Module to inference mode (this affects things like dropout)
        if not isinstance(source, torch.Tensor):
            source_input = torch.tensor(source).view(1, -1)
        else:
            source_input = source.view(1, -1)
        encoded_source,source_padding = self.encoder(source_input)
        batch_size = encoded_source.shape[0]
        input = torch.zeros((batch_size, 1)).to(torch.int64)
        final_candidates = []
        candidate_prob = []
        candidate_prob.append([0,input])
        while beam_size > 0:


            all_candidates = []

            for pair in candidate_prob:

                token = pair[1]

                
                y, attention_weights = self.decoder(encoded_source,source_padding,token)
                y = self.softmax(y)
 

                top_k_token = torch.topk(y[0][-1], beam_size)[1]
                top_k_prob = torch.topk(y[0][-1], beam_size)[0]


   


                for i in range (len(top_k_token)):
                    parent = pair[1]
                    new_sentence = torch.cat((parent,top_k_token[i].reshape(1,1)),dim=1)
                    parent_prob = pair[0]
                
                    new_prob = math.log(float(top_k_prob[i]))+ parent_prob
                    all_candidates.append([new_prob,new_sentence])
            
 
            # select top k
            all_candidates = all_candidates

            all_candidates.sort()
            new_candidates = all_candidates[len(all_candidates)-beam_size:]

            # then take a look on whether we should terminate
            p = 0

            while p < (len(new_candidates)):

                
                pair = new_candidates[p]
                sentence = pair[1]
                sentence = (sentence.tolist())[0]
                if sentence[-1] == 1:

                    pair[0] = pair[0]/len(sentence)
                    final_candidates.append(pair)
                    
                    
                    new_candidates.pop(p)
                    beam_size -=1
                elif len(sentence) >= max_length:

                    pair[0] = pair[0]/len(sentence)
                    final_candidates.append(pair)

                    new_candidates.pop(p)
                    beam_size -=1
                else:
                    p+=1
            candidate_prob = new_candidates

        final_candidates.sort()

        most_likely = final_candidates[-1]
        prob = most_likely[0]
        most_likely = most_likely[1]

        most_likely = most_likely.tolist()
        most_likely = most_likely[0]

        return (most_likely,prob)




        
        
        

def load_data() -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]]:
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/translation_data.bin', 'rb') as f:
        corpus, source_vocab, target_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(source_vocab), len(target_vocab), len(train_sentences),
                                              len(test_sentences)))
        return train_sentences, test_sentences, source_vocab, target_vocab

def preprocess_data(sentences: Tuple[List[int], List[int]], source_vocab_size,
                    target_vocab_size, max_length):
    
    source_sentences = []
    target_sentences = []

    for source, target in sentences:
        source = [0] + source + ([source_vocab_size] * (max_length - len(source) - 1))
        target = [0] + target + ([target_vocab_size] * (max_length - len(target) - 1))
        source_sentences.append(source)
        target_sentences.append(target)

    return torch.tensor(source_sentences), torch.tensor(target_sentences)

def decode_sentence(encoded_sentence: List[int], vocab: Dict) -> str:
    if isinstance(encoded_sentence, torch.Tensor):
        encoded_sentence = [w.item() for w in encoded_sentence]
    words = [vocab[w] for w in encoded_sentence if w != 0 and w != 1 and w in vocab]
    return " ".join(words)

def visualize_attention(source_sentence: List[int],
                        output_sentence: List[int],
                        source_vocab: Dict[int, str],
                        target_vocab: Dict[int, str],
                        attention_matrix: np.ndarray):
    """
    :param source_sentence_str: the source sentence, as a list of ints
    :param output_sentence_str: the target sentence, as a list of ints
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    source_length = 0
    while source_length < len(source_sentence) and source_sentence[source_length] != 1:
        source_length += 1

    target_length = 0
    while target_length < len(output_sentence) and output_sentence[target_length] != 1:
        target_length += 1

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix[:target_length, :source_length], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(source_length)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in source_vocab else source_vocab[x] for x in source_sentence[:source_length]]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(target_length)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in target_vocab else target_vocab[x] for x in output_sentence[:target_length]]))

    plt.show()
    plt.close()

def train(model: Transformer, train_source: torch.Tensor, train_target: torch.Tensor,
          test_source: torch.Tensor, test_target: torch.Tensor, target_vocab_size: int,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.0001):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(ignore_index=target_vocab_size)

    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)

    for ep in range(epochs):

        train_loss = 0
        test_loss = 0

        permutation = torch.randperm(train_source.shape[0])
        train_source = train_source[permutation]
        train_target = train_target[permutation]

        batches = train_source.shape[0] // batch_size
        model.train()
        for ba in tqdm(range(batches), desc=f"Epoch {ep + 1}"):

            optimizer.zero_grad()

            batch_source = train_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = train_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        test_batches = test_source.shape[0] // batch_size
        model.eval()
        for ba in tqdm(range(test_batches), desc="Test", leave=False):

            batch_source = test_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = test_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / batches
        epoch_test_loss[ep] = test_loss / test_batches
        print(f"Epoch {ep + 1}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")
    return epoch_train_loss, epoch_test_loss

def bleu_score(predicted: List[int], target: List[int], N: int = 4) -> float:
    """
    *** For students in 10-617 only ***
    (Students in 10-417, you can leave `raise NotImplementedError()`)

    Implement a function to compute the BLEU-N score of the predicted
    sentence with a single reference (target) sentence.

    Please refer to the handout for details.

    Make sure you strip the SOS (0), EOS (1), and padding (anything after EOS)
    from the predicted and target sentences.
    
    If the length of the predicted sentence or the target is less than N,
    the BLEU score is 0.
    """


    pre_start = predicted.index(0)
    if 1 in predicted:
        pre_end = predicted.index(1)
    else:
        pre_end = len(predicted)

    tar_start = target.index(0)
    if 1 in target:
        tar_end = target.index(1)
    else:
        tar_end  = len(tar_end)
    predicted = predicted[pre_start+1:pre_end]
    target = target[tar_start+1:tar_end]
    if len(predicted) < N:
        return 0

    # sliding
    bleu_score = 1
    for k in range (1,N+1):
        pre_gram = []
        tar_gram = []
        for i in range(0,len(predicted) - k+1):
            pre_gram.append(predicted[i:i+k])
        for i in range(0, len(target)-k+1):
            tar_gram.append(target[i:i+k])
        seen = set()
        count = 0
        for gram in pre_gram:
            if tuple(gram) not in seen:
                seen.add(tuple(gram))
                count += min(pre_gram.count(gram),tar_gram.count(gram))
        length = len(predicted)
        count = count/(length-k+1)
        bleu_score *= ((count)**(1.0/N))
    penalty = min(1,math.exp(1-len(target)/len(predicted)))
    bleu_score = bleu_score*penalty

    return bleu_score

    





if __name__ == "__main__":
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()
    
    train_source, train_target = preprocess_data(train_sentences, len(source_vocab), len(target_vocab), 12)
    test_source, test_target = preprocess_data(test_sentences, len(source_vocab), len(target_vocab), 12)
    print('finish loading')
    '''
    transformer = Transformer(len(train_source),len(train_target),256,1,1,1)
    train_loss, test_loss = train(transformer, train_source, train_target, test_source, test_target,len(target_vocab))
    torch.save(transformer.state_dict(),'m111.pkl')
    xlabel = list(range(0,30))
    plt.plot(np.array(xlabel),np.array(train_loss),label = 'train loss')
    plt.plot(np.array(xlabel),np.array(test_loss),label = 'test loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    leg = plt.legend(loc='upper right')
    plt.show()
    
    
    transformer = Transformer(len(train_source),len(train_target),256,2,4,3)

    transformer.load_state_dict(torch.load('m243.pkl'))
    first8_source = test_source[0:8,:]
    first8_target = test_target[0:8,:]

    print('load finished')
    print('////////////////////////')
    for i in range (8):
        source_sentence = first8_source[i,:]
        target_sentence = first8_target[i,:]
        print(f'sentence{i}:')
        print('source sentence is:')
        print(decode_sentence(source_sentence,source_vocab))
        print('target sentence is:')
        print(decode_sentence(target_sentence,target_vocab))
        predicted, prob = transformer.predict(source_sentence,beam_size = 3)
        print('predicted_sentence is:')
        print(decode_sentence(predicted,target_vocab))
        print('avg log-likelihood is:')
        print(prob)
        print('\n')
    

    transformer = Transformer(len(train_source),len(train_target),256,2,4,3)
    transformer.load_state_dict(torch.load('m243.pkl'))
    first3_source = train_source[0:3,:]
    first3_target = train_target[0:3,:]
    for i in range (3):
        source_sentence = first3_source[i,:]
        if not isinstance(source_sentence, torch.Tensor):
            source_input = torch.tensor(source_sentence).view(1, -1)
        else:
            source_input = source_sentence.view(1, -1)
        predicted, prob = transformer.predict(source_sentence,beam_size = 3)
        if not isinstance(predicted, torch.Tensor):
            predicted = torch.tensor(predicted).view(1, -1)
        else:
            predicted = predicted.view(1, -1)
        output,attention = transformer.forward(source_sentence,predicted)
        ssss


    transformer = Transformer(len(train_source),len(train_target),256,2,4,3)
    transformer.load_state_dict(torch.load('m243.pkl'))
    first100_source = test_source[0:100,:]
    first100_target = test_target[0:100,:]
    beam1 = 0
    beam2 = 0
    beam3 = 0
    beam4 = 0
    beam5 = 0
    beam6 = 0
    beam7 = 0
    beam8 = 0
    for i in range (100):
        print(i)
        source_sentence = first100_source[i,:]
        target_sentence = first100_target[i,:]
        predicted, prob = transformer.predict(source_sentence,beam_size = 1)
        beam1+=(prob)

        predicted, prob = transformer.predict(source_sentence,beam_size = 2)
        beam2+=(prob)

        predicted, prob = transformer.predict(source_sentence,beam_size = 3)
        beam3+=(prob)

        predicted, prob = transformer.predict(source_sentence,beam_size = 4)
        beam4+=(prob)

        predicted, prob = transformer.predict(source_sentence,beam_size = 5)
        beam5+=(prob)

        predicted, prob = transformer.predict(source_sentence,beam_size = 6)
        beam6+=(prob)

        predicted, prob = transformer.predict(source_sentence,beam_size = 7)
        beam7+=(prob)

        predicted, prob = transformer.predict(source_sentence,beam_size = 8)
        beam8+=(prob)
    x = list(range(1,9))
    y = [beam1/100,beam2/100,beam3/100,beam4/100,beam5/100,beam6/100,beam7/100,beam8/100]
    plt.plot(np.array(x),np.array(y))
    plt.xlabel("beam")
    plt.ylabel("avg log prob")
    leg = plt.legend(loc='upper right')
    plt.show()
    
    transformer111 = Transformer(len(train_source),len(train_target),256,1,1,1)
    transformer111.load_state_dict(torch.load('m111.pkl'))

    transformer113 = Transformer(len(train_source),len(train_target),256,1,1,3)
    transformer113.load_state_dict(torch.load('m113.pkl'))

    transformer221 = Transformer(len(train_source),len(train_target),256,2,2,1)
    transformer221.load_state_dict(torch.load('m221.pkl'))

    transformer223 = Transformer(len(train_source),len(train_target),256,2,2,3)
    transformer223.load_state_dict(torch.load('m223.pkl'))

    transformer243 = Transformer(len(train_source),len(train_target),256,2,4,3)
    transformer243.load_state_dict(torch.load('m243.pkl'))

    

    test_length = test_source.shape[0]
    for bs in range(1,5):
        bsocre = 0
        for i in range (test_length):
            source_sentence = test_source[i,:]
            source_sentence = source_sentence.tolist()

            target_sentence = test_target[i,:]
            target_sentence  = target_sentence .tolist()
            predicted, prob = transformer111.predict(source_sentence,beam_size = 3)
            bsocre += bleu_score(predicted, target_sentence , bs)
        print(bs)
        print(bsocre/test_length)
    print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

    for bs in range(1,5):
        bsocre = 0
        for i in range (test_length):
            source_sentence = test_source[i,:]
            source_sentence = source_sentence.tolist()

            target_sentence = test_target[i,:]
            target_sentence  = target_sentence .tolist()
            predicted, prob = transformer113.predict(source_sentence,beam_size = 3)
            bsocre += bleu_score(predicted, target_sentence , bs)
        print(bs)
        print(bsocre/test_length)
    print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

    for bs in range(1,5):
        bsocre = 0
        for i in range (test_length):
            source_sentence = test_source[i,:]
            source_sentence = source_sentence.tolist()

            target_sentence = test_target[i,:]
            target_sentence  = target_sentence .tolist()
            predicted, prob = transformer221.predict(source_sentence,beam_size = 3)
            bsocre += bleu_score(predicted, target_sentence , bs)
        print(bs)
        print(bsocre/test_length)
    print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

    for bs in range(1,5):
        bsocre = 0
        for i in range (test_length):
            source_sentence = test_source[i,:]
            source_sentence = source_sentence.tolist()

            target_sentence = test_target[i,:]
            target_sentence  = target_sentence .tolist()
            predicted, prob = transformer223.predict(source_sentence,beam_size = 3)
            bsocre += bleu_score(predicted, target_sentence , bs)
        print(bs)
        print(bsocre/test_length)
    print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')

    for bs in range(1,5):
        bsocre = 0
        for i in range (test_length):
            source_sentence = test_source[i,:]
            source_sentence = source_sentence.tolist()

            target_sentence = test_target[i,:]
            target_sentence  = target_sentence .tolist()
            predicted, prob = transformer243.predict(source_sentence,beam_size = 3)
            bsocre += bleu_score(predicted, target_sentence , bs)
        print(bs)
        print(bsocre/test_length)
    '''








    
   

        
        
        
        
        

        


    




    
    


