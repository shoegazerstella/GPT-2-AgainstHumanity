import gpt_2_simple as gpt2
import argparse


def main():

    sess = gpt2.start_tf_sess()

    gpt2.load_gpt2(sess)

    single_text = gpt2.generate(sess, return_as_list=True)[0]
    print(single_text)
    

if __name__ == "__main__":
    main()