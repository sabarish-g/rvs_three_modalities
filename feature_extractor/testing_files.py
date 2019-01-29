import pdb

#Train,dev,test images list
with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.trainImages.txt','r') as f:
    train_images = f.read().splitlines()
print('the total train images are: %s' %len(train_images))

with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.devImages.txt','r') as f:
    dev_images = f.read().splitlines()
print('the total dev images are: %s' %len(dev_images))

with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.testImages.txt','r') as f:
    test_images = f.read().splitlines()
print('the total test images are: %s' %len(test_images))   
#sorting all lists to keep them in order
train_images = sorted(train_images)
dev_images = sorted(dev_images)
test_images= sorted(test_images)


#getting all the wave files
with open('/shared/kgcoe-research/mil/new_cvs_data/flickr_audio/wav2capt.txt','r') as f:
    names = f.read().splitlines()    
audio_list = []
image_list = []
for index in names:
    audio_list.append(index.split(' ')[0])
    image_list.append(index.split(' ')[1]) 
print('the total audio files are: %s' %len(audio_list))    
print('the total images are: %s' %len(image_list))    

audio_list= sorted(audio_list)
image_list= sorted(image_list)

#Getting only the training images and audio from the entire audio list
index_list = [i for i,x in enumerate(image_list) if x in train_images]
print('the total train images are: %s' %len(index_list))

train_audio = []
for val in index_list:
    train_audio.append(audio_list[val])
print('the total audio files for training are: %s' %len(train_audio))

#Dealing with the captions
with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr8k.lemma.token.txt','r') as f:
    captions = f.read().splitlines()
file_list=[]
captions_list=[]
for line in captions:
    file_list.append(line.split('#')[0])
    sentence = line.split('#')[1].split('\t')[1]
    captions_list.append(sentence)

train_sentences = []
cap_files = []
for element in train_images:
    idx_list = [i for i,val in enumerate(file_list) if val==element]
    for i in idx_list:
        train_sentences.append(captions_list[i])
        cap_files.append(file_list[i])

print('the total captions files for training are: %s' %len(train_sentences))


pdb.set_trace()
