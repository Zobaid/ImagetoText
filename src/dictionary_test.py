from spellchecker import SpellChecker
import malaya
from symspellpy import SymSpell, Verbosity
import pytesseract
import cv2
import pkg_resources
spell = SpellChecker(distance=1)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
# prob_corrector = malaya.spell.symspell()
prob_corrector = malaya.spell.probability()

def symspell(word):
    # lookup suggestions for single-word input strings

    # max edit distance per lookup
    # (max_edit_distance_lookup <= max_dictionary_edit_distance)
    suggestions = sym_spell.lookup_compound(word,
                                            max_edit_distance=2)
    # display suggestion term, term, edit distance, frequency
    for suggestion in suggestions:
        # print("3) Symspell: Dictionary suggestion: ", suggestion)
        term, _, _ = str(suggestion).split(',')

    return term



def spellcheck(word):
    correct = spell.correction(word)
    return correct



def input_word(the_word, probability, fnImg):
    the_prob = probability
    recog_word = the_word


    misspelled = spell.unknown(recog_word)
    file = open('words.txt', 'a+')
    file.write(str(recog_word) +"\n")
    file.close()

    if len(misspelled) == 0:
        print("2) SpellChecker: Dictionary word same as recognized word by model")
    else:
        for word in misspelled:
            # Get the one `most likely` answer
            print("2) SpellChecker: Dictionary word is:", spell.correction(word))

        # Get a list of `likely` options
        # print("Potential dictionary word is:", spell.candidates(word))

    # lookup suggestions for single-word input strings
    input_term = (recog_word)
    # max edit distance per lookup
    # (max_edit_distance_lookup <= max_dictionary_edit_distance)
    suggestions = sym_spell.lookup_compound(recog_word,
                                            max_edit_distance=2)
    symspell_term = False
    # display suggestion term, term, edit distance, frequency
    for suggestion in suggestions:
        print("3) Symspell: Dictionary suggestion: ", suggestion)
        term, _, _ = str(suggestion).split(',')
        if recog_word == term:
            symspell_term = True
        else:
            symspell_term = False


    malaya_term = False
    try:
        malay_word = prob_corrector.correct(recog_word)
        print("4) Malaya dictionary: " ,malay_word)
        ORGINAL = cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE)
        print("5) pytesseract:", pytesseract.image_to_string(ORGINAL, lang='eng'))  ##
        if recog_word == malay_word:
            malaya_term = True
    except IndexError:
        pass



    if len(misspelled) == 0:
        return True, the_prob, symspell_term, malaya_term
    else:
        return False, the_prob, symspell_term, malaya_term


def tessract_test(img_path):
    img_cv = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # dic = pytesseract.image_to_data(img_rgb, lang='eng', output_type='data.frame')
    dic = pytesseract.image_to_data(img_rgb, lang='eng', output_type='dict')
    print("Pytessract tester: \n")  #
    print(pytesseract.image_to_string(img_rgb, lang='eng'))
    left_list = dic['left']
    top_list = dic['top']
    width_list = dic['width']
    height_list = dic['height']
    confident_list = dic['conf']
    text_list = dic['text']
    word_num_list = dic['word_num']
    line_num_list = dic['line_num']
    par_num_list = dic['par_num']
    # print(dic)
    # print(dic.iloc[5])
    # print(dic.iloc[6])
    # print(dic.iloc[7])
    filename = 'cho'
    newline = '\n'
    the_text = ""
    for i in range(len(confident_list)):
        if 90 > int(confident_list[i]) >= 0:
            print(word_num_list[i], left_list[i], top_list[i], width_list[i], height_list[i], confident_list[i], text_list[i])
            x = int(top_list[i])
            y = int(left_list[i])
            w = int(width_list[i])
            h = int(height_list[i])
            # cv2.rectangle(img_cv, (y, x), (y + w, x + h), (0, 0, 0), 3)
            image_to_show = img_cv[x:x + h, y:y + w]
            cv2.namedWindow('CROP IMAGE', cv2.WINDOW_NORMAL)
            cv2.imshow("CROP IMAGE", image_to_show)
            cv2.imwrite('../output_words/' + filename + '/%d.png' % i, image_to_show)  # save word
            cv2.waitKey()


    for i in range(len(confident_list)):
        print(word_num_list[i], par_num_list[i], line_num_list[i], left_list[i], top_list[i], width_list[i], height_list[i], confident_list[i], text_list[i])
        if word_num_list[i] == 0:
            the_text += newline
        else:
            the_text += str(text_list[i])+" "


