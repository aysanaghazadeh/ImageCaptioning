from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dropout, Dense, Embedding, LSTM, add, Input
from keras.utils import plot_model

from FeaturExtraction import VGG, ResNet, load_photo_features
from pickle import dump
from TextDataPreparation import save_descriptions, to_vocab, clean_descriptions, load_clean_descriptions, load_set, \
    create_tokenizer, create_sequences, max_length

directory = '/Users/aisanaghazade/PycharmProjects/ImageCaptioning/Flicker8k_Dataset'
filename_token = '/Users/aisanaghazade/PycharmProjects/ImageCaptioning/Flickr8k_text/Flickr8k.token.txt'
filename_trainImage = '/Users/aisanaghazade/PycharmProjects/ImageCaptioning/Flickr8k_text/Flickr_8k.trainImages.txt'



##since we have once extracted features and saved them to seperate files we do not call this function anymore

def feature_extraction(directory):
    # feature extraction using VGG16
    features = VGG(directory)
    print('Extracted Features: %d' % len(features))
    # save to file
    dump(features, open('VGGFeatures.pkl', 'wb'))

    # feature extraction using ResNet50
    features = ResNet(directory)
    print('Extracted Features: %d' % len(features))
    # save to file
    dump(features, open('ResNetFeatures.pkl', 'wb'))

# #since we have once cleaned text file and saved them to descriptions.txt we have commented next lines

# #text cleaning
def txt_cleaning(filename_token):
    vocabulary = to_vocab(filename_token)
    descriptions = clean_descriptions(filename_token)
    print('Vocabulary Size: %d' % len(vocabulary))
    save_descriptions(descriptions, 'descriptions.txt')


def define_model(vocab_size, max_length):
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	print(model.summary())
	# plot_model(model, to_file='model.png', show_shapes=True)
	return model


# train dataset

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('VGGfeatures.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)

# dev dataset

# load test set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('VGGfeatures.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)


# fit model

# define the model
model = define_model(vocab_size, max_length)
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
