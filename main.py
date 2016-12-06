from part_5 import *
from part_3 import decode_file as decode_file_3
from part_4 import top_m_decode_file as decode_file_4
from part_5 import decode_file as decode_file_5

from part_3 import add_predicted_symbols_to_file as write_to_file_3
from part_4 import write_part_4_dev_out as write_to_file_4
from part_5 import add_predicted_symbols_to_file as write_to_file_5

"""
Attaining dev.p3.out for each data set
"""
predicted_symbols_en3 = decode_file_3("data/EN/train", "data/EN/dev.in")
write_to_file_3(predicted_symbols_en3, "data/EN/dev.in", "output/EN/dev.p3.out")

predicted_symbols_es3 = decode_file_3("data/ES/train", "data/ES/dev.in")
write_to_file_3(predicted_symbols_es3, "data/ES/dev.in", "output/ES/dev.p3.out")

predicted_symbols_cn3 = decode_file_3("data/CN/train", "data/CN/dev.in")
write_to_file_3(predicted_symbols_cn3, "data/CN/dev.in", "output/CN/dev.p3.out")

predicted_symbols_sg3 = decode_file_3("data/SG/train", "data/SG/dev.in")
write_to_file_3(predicted_symbols_sg3, "data/SG/dev.in", "output/SG/dev.p3.out")

"""
Attaining dev.p4.out for each data set
"""
top_5_en4 = decode_file_4(5, "data/EN/train", "data/EN/dev.in")
write_to_file_4(top_5_en4, "data/EN/dev.in", "output/EN/dev.p4.out")

top_5_es4 = decode_file_4(5, "data/ES/train", "data/ES/dev.in")
write_to_file_4(top_5_es4, "data/ES/dev.in", "output/ES/dev.p4.out")

top_5_en4_processed = decode_file_4(5, "data/EN/train_processed", "data/EN/dev.in_processed")
write_to_file_4(top_5_en4_processed, "data/EN/dev.in", "output/EN/dev.p4_processed.out")

top_5_es4_processed = decode_file_4(5, "data/ES/train_processed", "data/ES/dev.in_processed")
write_to_file_4(top_5_es4_processed, "data/ES/dev.in", "output/ES/dev.p4_processed.out")

"""
Attaining dev.p5.out for each data set
"""
predicted_symbols_en5 = decode_file_5("data/EN/train_processed", "data/EN/dev.in_processed")
write_to_file_5(predicted_symbols_en5, "data/EN/dev.in", "output/EN/dev.p5.out")

predicted_symbols_es5 = decode_file_5("data/ES/train_processed", "data/ES/dev.in_processed")
write_to_file_5(predicted_symbols_es5, "data/ES/dev.in", "output/ES/dev.p5.out")

predicted_symbols_en5 = decode_file_5("data/EN/train_processed", "data/p5_test/EN/test.in_processed")
write_to_file_5(predicted_symbols_en5, "data/p5_test/EN/test.in", "output/EN/dev.p5.out")

predicted_symbols_es5 = decode_file_5("data/ES/train_processed", "data/p5_test/ES/test.in_processed")
write_to_file_5(predicted_symbols_es5, "data/p5_test/ES/test.in", "output/ES/dev.p5.out")