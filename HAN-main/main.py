import random
from modules.DecoderModel import *
from driver.Config import *
from driver.Classifier import *
import pickle
from modules.LSTMEncoderModel import *
import time
from modules.CHANEncoderModel import *
from driver.TrainTest import train
from data.Dataloader import read_corpus,read_instance


def evaluate(data, classifier, vocab, outputFile):
    start = time.time()
    classifier.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    label_correct, overall = 0, 0
    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, masks, labels = \
            batch_data_variable(onebatch, vocab)
        pred_labels = classifier.predict(words, extwords, masks)
        write_sents(pred_labels, onebatch, vocab, output)
        true_labels = to_tensor(labels)
        label_correct += pred_labels.eq(true_labels).cpu().sum().item()
        overall += len(labels)
    output.close()
    acc = label_correct * 100.0 / overall
    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  classifier time = %.2f " % (len(data), during_time))
    return label_correct, overall, acc


def write_sents(pred_labels, onebatch, vocab, file):
    label_strs = []
    for label in pred_labels:
        l = label.item()
        pred_label_str = vocab.id2label(l)
        label_strs.append(pred_label_str)

    max_len = len(onebatch)
    assert max_len == len(label_strs)
    for idx in range(max_len):
        file.write(label_strs[idx] + "$#$")
        for word in onebatch[idx][1]:
            file.write(word + " ")
        file.write("\n")


random.seed(666)
np.random.seed(666)
torch.cuda.manual_seed(666)
torch.manual_seed(666)


### gpu
gpu = torch.cuda.is_available()
print("GPU available: ", gpu)
print("CuDNN: \n", torch.backends.cudnn.enabled)

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_file', default='new-classifer-model/config.cfg')
argparser.add_argument('--model', default='BaseParser')
argparser.add_argument('--thread', default=4, type=int, help='thread num')
argparser.add_argument('--use-cuda', action='store_true', default=True)

args, extra_args = argparser.parse_known_args()
config = Configurable(args.config_file, extra_args)

vocab = creatVocab(config)

print(vocab._id2role)

vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

args, extra_args = argparser.parse_known_args()
config = Configurable(args.config_file, extra_args)
torch.set_num_threads(args.thread)

config.use_cuda = True
if gpu and args.use_cuda: config.use_cuda = True
print("\nGPU using status: ", config.use_cuda)

# print(config.use_cuda)

encoder = eval(config.encoder)(vocab, config, vec)
decoder = Decoder(vocab, config)
print(encoder)
print(decoder)

if config.use_cuda:
    torch.backends.cudnn.enabled = True
    encoder = encoder.cuda()
    decoder = decoder.cuda()

    # parser = parser.cuda()

classifier = Classifier(encoder, decoder)

train_data = read_corpus(config.train_file, config.max_sent_length, config.max_turn_length, vocab)
dev_data = read_corpus(config.dev_file, config.max_sent_length, config.max_turn_length, vocab)
test_data = read_corpus(config.test_file, config.max_sent_length, config.max_turn_length, vocab)

print("train num:", len(train_data))
print("dev num:", len(dev_data))
print("test num:", len(test_data))

train(train_data, dev_data, test_data, classifier, vocab, config)