import numpy as np 
import tensorflow as tf 
import pickle 
import os 
import argparse
from model import *   
import random
from prepro import *
import time
import pandas as pd
from bleu_eval import *

def map_back_to_word(predict,index2word):
    
    batch_seq = []
    for data in range(predict.shape[0]):
        sequence = ''
        for index in predict[data]:
            word=index2word[index[0]]
            if word in [ '<bos>','<pad>','<unk>']:
                word = ''
            word_ = word + ' '   
            if word in ['<eos>']:
                word_ = ''
            
            sequence += word_
        batch_seq += [sequence]
    #print('batch_seq=',batch_seq)
    return batch_seq




    



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    parser = argparse.ArgumentParser()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True

    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--rnn_dim', type=int, default=1024, help='rnn_dim')
    parser.add_argument('--num_layer',type=int,default=2,help='number of layer' )
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--outf', default='./new/', help='folder to output images and model checkpoints')
    parser.add_argument('--epoch',type=int,default=200)
    parser.add_argument('--keep_prob',type=float,default=0.5)
    parser.add_argument('--encoding_embedding_size',type=int,default=1024,help='encoding_embedding_size')
    parser.add_argument('--decoding_embedding_size',type=int,default=1024,help='decoding_embedding_size')
    parser.add_argument('--max_encoder_steps',type=int, default=64,help='max_encoder_steps')
    parser.add_argument('--max_decoder_steps',type=int,default=25,help='max_decoder_steps')
    parser.add_argument('--dim_video_feat',type=int,default=4096, help='dim_video_feat')
    parser.add_argument('--dim_video_frame',type=int,default=80,help='dim_video_frame')
    parser.add_argument('--sample_data',type=int,default=1280,help='sample_data')
    parser.add_argument('--test_video_feat_folder',type=str,default="../dataset/MLDS_hw2_1_data/testing_data/feat/")
    parser.add_argument("--testing_label_json_file",default="../dataset/MLDS_hw2_1_data/testing_label.json")
    parser.add_argument("--attention_method",type=bool,default=True,help='Use attention mechanism ')
    parser.add_argument("--beam_search",type=bool,default=False,help='beam_search')
    parser.add_argument("--beam_size",type=int,default=3,help='beam_search')
    parser.add_argument("--output_testset_filename",type=str,default='output.csv')



    opt = parser.parse_args()

    word2index = pickle.load(open('word2index.p','rb'))
    index2word = pickle.load(open('index2word.p','rb'))
    video_caption_dict = pickle.load(open('video_caption_dict.p','rb'))
    video_feat_dict    = pickle.load(open('video_feat_dict.p','rb'))
    video_IDs          = pickle.load(open('video_IDs.p','rb'))

    print ('Reading testing files...')
    test_video_feat_filenames = os.listdir(opt.test_video_feat_folder)
    test_video_feat_filepaths = [(opt.test_video_feat_folder + filename) for filename in test_video_feat_filenames]
    
    # Remove '.avi' from filename.
    test_video_IDs = [filename[:-4] for filename in test_video_feat_filenames]

    test_video_feat_dict = {}
    for filepath in test_video_feat_filepaths:
        test_video_feat = np.load(filepath)
        
        sampled_video_frame = sorted(random.sample(range(opt.dim_video_frame), opt.max_encoder_steps))
        test_video_feat = test_video_feat[sampled_video_frame]

        test_video_ID = filepath[: -4].replace(opt.test_video_feat_folder, "")
        test_video_feat_dict[test_video_ID] = test_video_feat
    
            
    #test_features = [ (file[:-4],np.load(opt.test_video_feat_folder + file)) for file in test_video_feat_filenames]
    
    test_video_caption = json.load(open(opt.testing_label_json_file, 'r'))
    

    target_vocab_to_int=word2index
    
    # opt.batch_size = opt.batch_size if not opt.beam_search else opt.batch_size * opt.beam_size

    Graph = tf.Graph()
    with Graph.as_default():
        inputs, targets, target_sequence_length, max_target_sequence_length,input_video_number =enc_dec_model_inputs()
        lr, keep_prob = hyperparam_inputs()
        (train_logits, inference_logits, inference_states)= seq2seq_model(inputs,
                                                targets,
                                                keep_prob,
                                                opt.batch_size,input_video_number,
                                                target_sequence_length,
                                                opt.dim_video_feat,
                                                max_target_sequence_length,
                                                len(target_vocab_to_int),
                                                opt.encoding_embedding_size,
                                                opt.decoding_embedding_size,
                                                opt.rnn_dim,
                                                opt.num_layer,
                                                target_vocab_to_int,
                                                opt.max_encoder_steps,beam_size=opt.beam_size,attention_mode=opt.attention_method,beam_search_mode=opt.beam_search)
        #print('train_logits=',train_logits)
        #print('inference_logits=',inference_logits)
        #print('inference_states=',inference_states)
        training_logits  = tf.identity(train_logits.rnn_output, name='logits')
        if opt.beam_search:
            inference_ids = inference_logits.predicted_ids
            inference_logits_=inference_states.log_probs
        else:
            inference_ids= tf.expand_dims(inference_logits.sample_id, -1,name='predictions')
            inference_logits_ = inference_logits.rnn_output
    
        # - Returns a mask tensor representing the first N positions of each cell.
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')


        with tf.name_scope("optimization"):
            # Loss function - weighted softmax cross entropy
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)
            
            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)
            
            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
        

    with tf.Session(graph=Graph) as sess:
        
        summary_writer = tf.summary.FileWriter(opt.outf, graph=Graph)
        tf.summary.scalar('loss', cost)
        summary_op = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        step = 0 

        if len(video_IDs) % 128 ==0:
            pass
        else:
            dummy=len(video_IDs) % 128
            video_IDs = video_IDs[:-dummy]
        time_st = time.time()
        for epoch in range(opt.epoch):
            
            sampled_feat_caption = list()
            
            #prepared for each data generation
            for ID in video_IDs:
                sampled_caption = random.sample(video_caption_dict[ID], 1)[0]
                sampled_video_frame = sorted(random.sample(range(opt.dim_video_frame), opt.max_encoder_steps))
                sampled_video_feat = video_feat_dict[ID][sampled_video_frame]
                sampled_feat_caption.append((sampled_video_feat, sampled_caption))
                
            random.shuffle(sampled_feat_caption)

            #generate dataset from above
            
            for number in range(0,opt.sample_data,opt.batch_size):
                
                batch_sample_feat_caption = sampled_feat_caption[number: number + opt.batch_size]
                batch_video_feats = [elements[0] for elements in batch_sample_feat_caption]
                batch_video_frame = [opt.max_decoder_steps] * opt.batch_size
                batch_captions = np.array(["<bos> "+ elements[1] for elements in batch_sample_feat_caption])

                # delete the member of sentence which over the fixed length and pad eos to sentence which length smaller than fix length 
                for index, caption in enumerate(batch_captions):
                    caption_words = caption.lower().split(" ")
                    if len(caption_words) < opt.max_decoder_steps:
                        batch_captions[index] = batch_captions[index] + " <eos>"
                    else:
                        new_caption = ""
                        for i in range(opt.max_decoder_steps - 1):
                            new_caption = new_caption + caption_words[i] + " "
                        batch_captions[index] = new_caption + "<eos>"

                #generate the list of number to represent the caption
                batch_captions_words_index = []
                for caption in batch_captions:
                    words_index = []
                    for caption_words in caption.lower().split(' '):
                        if caption_words in word2index:
                            words_index.append(word2index[caption_words])
                        else:
                            words_index.append(word2index['<unk>'])
                    batch_captions_words_index.append(words_index)

                batch_captions_matrix = pad_sequences(batch_captions_words_index, padding='post', maxlen=opt.max_decoder_steps)
                batch_captions_length = [len(x) for x in batch_captions_matrix]
                
                _, loss = sess.run(
                    [train_op, cost],
                    {inputs: batch_video_feats,
                    input_video_number: opt.batch_size,
                    targets: batch_captions_matrix,
                    lr: opt.lr,
                    target_sequence_length: batch_captions_length,
                    keep_prob: opt.keep_prob})

                step+=1

                

                if (step)%20 ==0:
                    time_end=time.time()
                    print('loss= %.4f,step = %4d epoch= %2d, number= %2d' %(loss,step,epoch,number))
                    print('It costs  %3d seconds'%(time_end-time_st))
                    time_st = time_end
                    get_summary=sess.run(summary_op,{inputs: batch_video_feats,
                    targets: batch_captions_matrix,
                    lr: opt.lr,
                    target_sequence_length: batch_captions_length,
                    keep_prob: opt.keep_prob})
                    summary_writer.add_summary(get_summary, epoch)

            if (epoch+1) % 10 == 0 :

                if len(test_video_IDs) % opt.batch_size != 0:
                    dummy = test_video_IDs[-(len(test_video_IDs) % opt.batch_size):]
                    test_video_IDs = test_video_IDs[:-(len(test_video_IDs) % opt.batch_size)]
                    
                f = open('evaluate_epoch_'+str(epoch+1)+'_txt.txt','w')
                for i in range(0,len(test_video_IDs),opt.batch_size):
                    batch_sample_ID = test_video_IDs[i:i+opt.batch_size]
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sample_ID]
                    scores,predict=sess.run([inference_logits_,inference_ids],feed_dict={inputs: batch_video_feats,keep_prob:1,target_sequence_length:[opt.max_decoder_steps]*opt.batch_size})
                    if opt.beam_search:
                        batch_sequence=map_back_to_word(predict,index2word)
                    else:
                        batch_sequence=map_back_to_word(predict,index2word)
                    disp=zip(batch_sample_ID,batch_sequence)
                    for item in disp:
                        f.write(str(item[0])+ '    ' +str(item[1])+ '\n')
                f.close()

            if (epoch+1) % 2 == 0:
                test_video_Caption = []
                if len(test_video_IDs) % opt.batch_size != 0:
                    dummy = test_video_IDs[-(len(test_video_IDs) % opt.batch_size):]
                    test_video_IDs_ = test_video_IDs[:-(len(test_video_IDs) % opt.batch_size)]
                    test_video_IDs_ += test_video_IDs[-opt.batch_size:]
                for i in range(0,len(test_video_IDs_),opt.batch_size):
                    batch_sample_ID = test_video_IDs_[i:i+opt.batch_size]
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sample_ID]
                    scores,predict=sess.run([inference_logits_,inference_ids],feed_dict={inputs: batch_video_feats,keep_prob:1,target_sequence_length:[opt.max_decoder_steps]*opt.batch_size})
                    #print('predict_result=',predict.shape)
                    #print('scores=',scores)
                    if opt.beam_search:
                        batch_sequence=map_back_to_word(predict,index2word)
                    else:
                        batch_sequence=map_back_to_word(predict,index2word)
                    disp=zip(batch_sample_ID,batch_sequence)
                    for item in disp:
                        test_video_Caption.append(item[1])
                    
                print('ids',test_video_IDs_)
                print('captions',test_video_Caption)
                
                df = pd.DataFrame(np.array([test_video_IDs_, test_video_Caption]).T)
                df.to_csv(opt.output_testset_filename, index=False, header=False)
                result = {}
                with open(opt.output_testset_filename, 'r') as f:
                    for line in f:
                        line = line.rstrip()
                        test_id, caption = line.split(',')
                        result[test_id] = caption
                        
                bleu=[]
                for item in test_video_caption:
                    score_per_video = []
                    captions = [x.rstrip('.') for x in item['caption']]
                    #print('item[id]',item['id'])
                    score_per_video.append(BLEU(result[item['id']],captions,True))
                    bleu.append(score_per_video[0])

                average = sum(bleu) / len(bleu)
                print("Average bleu score is " + str(average))








            
                    




                




                
            



            








    

    

    



