from spellchecker import SpellChecker
import malaya
from symspellpy import SymSpell, Verbosity
import pytesseract
import cv2
import pkg_resources
import main
spell = SpellChecker()
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


def tessract_test(img_path, filename):
    img_cv = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # dic = pytesseract.image_to_data(img_rgb, lang='eng', output_type='data.frame')
    dic = pytesseract.image_to_data(img_rgb, lang='eng', output_type='dict')
    print("Recognition with pyTesseract only")
    print("PYtess string output = " +pytesseract.image_to_string(img_rgb, lang='eng'))
    left_list = dic['left']
    top_list = dic['top']
    width_list = dic['width']
    height_list = dic['height']
    confident_list = dic['conf']
    text_list = dic['text']
    word_num_list = dic['word_num']

    filename_output = filename
    for i in range(len(confident_list)):
        if 90 > int(confident_list[i]) >= 0:
            # print(word_num_list[i], left_list[i], top_list[i], width_list[i], height_list[i], confident_list[i], text_list[i])
            x = int(top_list[i])
            y = int(left_list[i])
            w = int(width_list[i])
            h = int(height_list[i])
            # cv2.rectangle(img_cv, (y, x), (y + w, x + h), (0, 0, 0), 3)
            image_to_show = img_cv[x:x + h, y:y + w]
            # cv2.namedWindow('CROP IMAGE', cv2.WINDOW_NORMAL)
            # cv2.imshow("CROP IMAGE", image_to_show)
            cv2.imwrite('../output_words/' + filename_output + '/%d.png' % i, image_to_show)  # save word
            # cv2.waitKey()
    return word_num_list, text_list, confident_list
