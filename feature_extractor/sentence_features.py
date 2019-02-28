import numpy as np
import pdb
import os
import argparse
import skipthoughts


def main(args):
    #Dealing with the captions
    with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr8k.lemma.token.txt','r') as f:
        captions = f.read().splitlines()
    file_list=[]
    captions_list=[]
    for line in captions:
        file_list.append(line.split('#')[0])
        sentence = line.split('#')[1].split('\t')[1]
        captions_list.append(sentence)
    
    sentences = []
    cap_files = []    
    
    
    if args.mode == 'train':
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.trainImages.txt','r') as f:
            train_images = f.read().splitlines()
        print('the total train images are: %s' %len(train_images))
        train_images = sorted(train_images)
        for element in train_images:
            idx_list = [i for i,val in enumerate(file_list) if val==element]
            for i in idx_list:
                sentences.append(captions_list[i])
                cap_files.append(file_list[i])
        print('the total captions files for train are: %s' %len(sentences))
       
    elif args.mode == 'dev':
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.devImages.txt','r') as f:
            dev_images = f.read().splitlines()
        print('the total train images are: %s' %len(dev_images))
        dev_images = sorted(dev_images)
        for element in dev_images:
            idx_list = [i for i,val in enumerate(file_list) if val==element]
            for i in idx_list:
                sentences.append(captions_list[i])
                cap_files.append(file_list[i])
        print('the total captions files for dev are: %s' %len(sentences))

    elif args.mode == 'test':
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.testImages.txt','r') as f:
            test_images = f.read().splitlines()
        print('the total train images are: %s' %len(test_images))
        test_images = sorted(test_images)
        for element in test_images:
            idx_list = [i for i,val in enumerate(file_list) if val==element]
            for i in idx_list:
                sentences.append(captions_list[i])
                cap_files.append(file_list[i])
        print('the total captions files for test are: %s' %len(sentences))
    
    else:
        print('incorrect mode selected')
    all_text_data = np.array([])
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    vectors = encoder.encode(sentences)
    
    if args.mode == 'train':
        np.save(os.path.join(args.save_path,'flickr8k_train_sentence_features.npy'), vectors)
    if args.mode == 'dev':
        np.save(os.path.join(args.save_path,'flickr8k_dev_sentence_features.npy'), vectors)
    if args.mode == 'test':
        np.save(os.path.join(args.save_path,'flickr8k_test_sentence_features.npy'), vectors)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--save_path', type = str, default = '/shared/kgcoe-research/mil/new_cvs_data/sentence_features', help = 'path to save the features')
    parser.add_argument('--mode', type=str, default='train', help='Feature extraction for which phase?')
    args=parser.parse_args()
    main(args)
    