from __future__ import division
from __future__ import print_function
from flask import Flask, render_template, request
from waitress import serve
from src.Image_Processing import Image_Processing
import argparse
#import editdistance
from src.DataLoader import DataLoader, Batch
from src.SamplePreprocessor import preprocess
#from src.Model import Model, DecoderType
#import cv2
import os
#import src.word_seg
#import src.dictionary_test
#import src.tess
import flask
import sys
from src.Image_Processing import Image_Processing



app = Flask(__name__)
app.config['DEBUG'] == True

class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnCorpus = '../data/corpus.txt'


def getint(name):
    num, _ = name.split('.')
    return int(num)

def train(model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 5  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate


def infer(model, fnImg):
    "recognize text in image provided by file path"

    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)

    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    return recognized[0], probability[0]




def run(filename):
    "main function"
    # optional command line args

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

    # infer text on test image
    else:
        index_list = []
        result_list = []
        prob_list = []
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)

        for dirpath, dirnames, files in os.walk('../output_words/' + filename, topdown=False):
            for sub_file in sorted(files, key=getint):
                img_path = dirpath + '/' + sub_file
                # print('---------------------------------------------------')
                index_number, _ = str(sub_file).split('.')
                # print("File path: "+img_path)
                try:
                    result, prob = infer(model, img_path)
                except ValueError:
                    print("Value error")
                    continue
                # print(index_number, result, prob)
                index_list.append(index_number)
                result_list.append(result)
                prob_list.append(prob)

        return index_list, result_list, prob_list







#---Define routes---
#Home route
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

#htrengine route
@app.route('/htrengine', methods=['POST'])
def htrengine():
    try:
        #file = request.files['img']
        #return file.filename
        the_filename = ""
        for dirpath, dirnames, files in os.walk('input_words/', topdown=False):
            for sub_file in sorted(files):
                img_path = dirpath + sub_file
                the_filename, _ = sub_file.split('.')
                #try:
                #   os.mkdir('output_words/' + the_filename)
                #except FileExistsError:
                #   print("Cannot read image because file exist already")
                    #os.mkdir('../output_words/' + the_filename+1)
                #  continue
                word_num, word_text, conf_word = tess.tessract_test(img_path, the_filename)  # this will produce output and word segmentation
                index, result, prob = run(the_filename)  # NN engine can only run once, need to refactor to place it out of this loop
                # Comparison of score between pytesseract model & NN mdel
                for i in range(len(index)):
                    if int(conf_word[int(index[i])]) > 0:
                        score = int(conf_word[int(index[i])])/100
                    else:
                        score = int(conf_word[int(index[i])])
                    # print(float(prob[i]),float(score))
                    # if probabilty or model higher than pytesseract use, model!
                    if float(prob[i]) > float(score):
                        print("REPLACE prob comparison",  word_text[int(index[i])],">", result[i])
                        word_text[int(index[i])] = result[i]
                    else:
                        # Feed the pytesseract word into dictionary(symspell)
                        pyword = word_text[int(index[i])]
                        word_correct = dictionary_test.spellcheck(pyword)  # can adjust to different dictionery
                        print("REPLACE with dictionary",  word_text[int(index[i])],">", word_correct)
                        word_text[int(index[i])] = word_correct        
                # Tabulation of text
                the_text = ""
                newline = '\n'
                for i in range(len(word_num)):
                    if word_num[i] == 0:
                        the_text += newline
                    else:
                        the_text += str(word_text[i]) + " "
                print("Correction with handwriting model & dictionary")
                print(the_text)
                print("Working!")
                #return the_text
    except:
        return sys.exc_info()[0]

            



serve(app, host='0.0.0.0', port=700) 

    
if __name__ == '__main__':
    #htrengine()
    print("App is running!")

            