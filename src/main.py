from __future__ import division
from __future__ import print_function
import argparse
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import cv2
import os
import word_seg
import dictionary_test


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/sentence_2/0.png'
    fnInfer1 = '../data/sentence_2/1.png'
    fnInfer2 = '../data/sentence_2/2.png'
    fnInfer3 = '../data/sentence_2/3.png'
    fnInfer4 = '../data/sentence_2/4.png'
    fnInfer5 = '../data/sentence_2/5.png'

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
    print('1) Model: Recognized:', '"' + recognized[0] + '"' + ' Probability:', probability[0])

    output = dictionary_test.input_word(recognized[0], probability[0], fnImg)
    return output



def main(filename):
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

        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)

        spell_checker_match = 0
        spell_checker_symspell_match = 0
        spell_checker_malaya = 0
        num_files = 0
        model_probability = 0
        for dirpath, dirnames, files in os.walk('../output_words/' + filename, topdown=False):
            for sub_file in sorted(files, key=getint):
                num_files += 1
                img_path = dirpath + '/' + sub_file
                print('---------------------------------------------------')
                print("File path: "+img_path)
                try:
                    result, probability, result2, result3 = infer(model, img_path)
                except ValueError:
                    print("Encounter value error, skip this image")
                    continue
                model_probability += probability
                if result is True:
                    spell_checker_match += 1
                if result2 is True:
                    spell_checker_symspell_match += 1
                if result3 is True:
                    spell_checker_malaya += 1


        print('---------------SUMMARY RESULT of image:', filename+' ---------------')
        percentage_correct = (spell_checker_match/num_files) * 100
        percentage_correct_2 = (spell_checker_symspell_match/num_files) * 100
        percentage_correct_3 = (spell_checker_malaya/num_files) * 100
        performance_model = (model_probability/num_files) * 100
        print('Model performance probability percentage:', performance_model)
        print('Total number words:', num_files)
        print('Total number words correct based on dictionary(SpellChecker):', spell_checker_match)
        print('Percentage correctly recognised based on dictionary(SpellChecker):', percentage_correct)
        print('Total number words correct based on dictionary(Symspell):', spell_checker_symspell_match)
        print('Percentage correctly recognised based on dictionary(Symspell):', percentage_correct_2)
        print('Total number words correct based on dictionary(Malaya):', spell_checker_malaya)
        print('Percentage correctly recognised based on dictionary(Malaya):', percentage_correct_3)



if __name__ == '__main__':


    the_filename = ""
    for dirpath, dirnames, files in os.walk('../input_words/', topdown=False):
        for sub_file in sorted(files):
            img_path = dirpath + sub_file
            the_filename, _ = sub_file.split('.')
            os.mkdir('../output_words/' + the_filename)
            # dictionary_test.tessract_test(img_path)
            word_seg.word_segmentation(img_path, the_filename)

    main(the_filename)  # NN model only runs for the last image that was read. Need to refactor code to read all
