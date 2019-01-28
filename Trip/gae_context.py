# -*- coding:  UTF-8 -*-
from __future__ import division
import math
from tensorflow.python.layers.core import Dense
import seq2seq_c as seqc
from metric import *
from ops import *
import time
import datetime
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
# =============================== vars ====================================== #
is_initial=True
EPSILON = 1e-6
batch_size=8
n_hidden=512
AE_learning_rate=0.6
initializer=tf.truncated_normal_initializer(stddev=0.02)
critic_lr=1e-02
gen_lr=1e-02
z_dim=128
c_dim=512
train_iters =20   # 遍历样本次数 for training
embedding_size=256
dynamic_traning=True
dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb','TKY_split200']
dat_ix=5
poi_name="poi-"+dat_suffix[dat_ix]+".csv" #Edin
tra_name="traj-"+dat_suffix[dat_ix]+".csv"
embedding_name=dat_suffix[dat_ix]
model='./logs/model_'+embedding_name+'.pkt'
# =============================== data load ====================================== #
#load original data
op_tdata = open('origin_data/'+poi_name, 'r')

ot_tdata = open('origin_data/'+tra_name, 'r')
print 'To Train',dat_suffix[dat_ix]
POIs=[]
Trajectory=[]
for line in op_tdata.readlines():
    lineArr = line.split(',')
    temp_line=list()
    for item in lineArr:
        temp_line.append(item.strip('\n'))
    POIs.append(temp_line)
POIs=POIs[1:] #remove first line

def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088 # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist =  2 * radius * np.arcsin( np.sqrt(
                (np.sin(0.5*dlat))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5*dlng))**2 ))
    return dist
#



#print POIs
get_POIs={}
char_pois=[] #pois chars

for items in POIs:
    char_pois.append(items[0])
    get_POIs.setdefault(items[0],[]).append([items[2],items[3]]) # pois to category
Users=[]
poi_count={}
for line in ot_tdata.readlines():
    lineArr=line.split(',')
    temp_line=list()
    if lineArr[0]=='userID':
        continue
    poi_count.setdefault(lineArr[2], []).append(lineArr[2])
    for i in range(len(lineArr)):

        if i==0:
            user = lineArr[i]
            Users.append(user)  # add user id
            temp_line.append(user)
            continue
        temp_line.append(lineArr[i].strip('\n'))
    Trajectory.append(temp_line)
Users=sorted(list(set(Users)))
print 'user number',len(Users)

TRAIN_TRA=[]
TRAIN_USER=[]
TRAIN_TIME=[]
TRAIN_DIST=[]
DATA={} #temp_data
for index in range(len(Trajectory)):
    if(int(Trajectory[index][-2])>=3): #the length of the trajectory must over than 3
        DATA.setdefault(Trajectory[index][0]+'-'+Trajectory[index][1],[]).append([Trajectory[index][2],Trajectory[index][3],Trajectory[index][4]]) #userID+trajID



# #calc distance
# distance_count=[]
# for i in range(len(POIs)):
#     lon1=float(POIs[i][2])
#     lat1=float(POIs[i][3])
#     for j in range(i+1,len(POIs)):
#         lon2 = float(POIs[j][2])
#         lat2 =float( POIs[j][3])
#         dist=calc_dist_vec(lon1,lat1,lon2,lat2)
#         distance_count.append(dist)
# print 'max',max(distance_count)
# print 'min',min(distance_count)

#calc_distance
distance_count=[]
for key in DATA.keys():
    traj=DATA[key]
    #print traj
    for i in range(len(traj)):
        #print get_POIs[traj[i][0]][0][0]
        lon1=float(get_POIs[traj[i][0]][0][0])
        lat1=float(get_POIs[traj[i][0]][0][1])
        for j in range(i+1,len(traj)):
            lon2 = float(get_POIs[traj[j][0]][0][0])
            lat2 = float(get_POIs[traj[j][0]][0][1])
            distance_count.append(calc_dist_vec(lon1,lat1,lon2,lat2))
upper_dis=max(distance_count)
lower_dis=min(distance_count)
print len(distance_count)
# rand_l=np.random.rand(-1,1)
# rand_u=np.random.rand(-1,1)
# print rand_l
# val=((min(distance_count)-min(distance_count))*rand_l+(max(distance_count)-min(distance_count))*rand_u)/(min(distance_count)-min(distance_count)+max(distance_count)-min(distance_count))
# # def calc_distance(distance):
# #     val = ((distance - min(distance_count)) * rand_l + (max(distance_count) - distance) * rand_u)/(distance - min(distance_count) + max(distance_count) - distance)
# #     return val

for keys in DATA.keys():
    user_traj=DATA[keys]
    temp_poi=[]
    temp_time=[]
    temp_dist=[]
    for i in range(len(user_traj)):
        temp_poi.append(user_traj[i][0]) #add poi id
        lon1=float(get_POIs[user_traj[i][0]][0][0])
        lat1=float(get_POIs[user_traj[i][0]][0][1])
        lons=float(get_POIs[user_traj[0][0]][0][0])
        lats=float(get_POIs[user_traj[0][0]][0][1])
        lone=float(get_POIs[user_traj[-1][0]][0][0])
        late=float(get_POIs[user_traj[-1][0]][0][1])
        sd=calc_dist_vec(lon1,lat1,lons,lats)
        ed = calc_dist_vec(lon1, lat1, lone, late)
        value1=0.5*(sd)/max(distance_count)
        value2=0.5*(ed)/max(distance_count)
        #print value
        temp_dist.append([value1,value2]) #lon,lat

        dt = time.strftime("%H:%M:%S", time.localtime(int(user_traj[i][1:][0])))
        #print dt.split(":")[0]
        temp_time.append(int(dt.split(":")[0])) #add poi time
    TRAIN_USER.append(keys)
    TRAIN_TRA.append(temp_poi)
    TRAIN_TIME.append(temp_time)
    TRAIN_DIST.append(temp_dist)
dictionary={}
for key in poi_count.keys():
    count=len(poi_count[key])
    dictionary[key]=count
dictionary['GO']=1
dictionary['PAD']=1
dictionary['END']=1
new_dict=sorted(dictionary.items(),key = lambda x:x[1],reverse = True)

print 'poi number is',len(new_dict)-3
voc_poi=list()

for item in new_dict:
    voc_poi.append(item[0]) #has been sorted by frequency

def extract_words_vocab():
    int_to_vocab = {idx: word for idx, word in enumerate(voc_poi)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
int_to_vocab, vocab_to_int=extract_words_vocab()

#generate pre-traning dataset
new_trainT = list()
for i in range(len(TRAIN_TRA)): #TRAIN
    temp = list()
    temp.append(vocab_to_int['GO'])
    for j in range(len(TRAIN_TRA[i])):
        temp.append(vocab_to_int[TRAIN_TRA[i][j]])
    temp.append(vocab_to_int['END'])
    temp.append(vocab_to_int['PAD'])
    new_trainT.append(temp)

#generate traning dataset
new_trainTs = list()
for i in range(len(TRAIN_TRA)): #TRAIN
    temp = list()
    for j in range(len(TRAIN_TRA[i])):
        temp.append(vocab_to_int[TRAIN_TRA[i][j]])
    new_trainTs.append(temp)

dataset=open('data/'+embedding_name+'_set.dat','w')
for i in range(len(new_trainTs)):
    dataset.write(str(TRAIN_USER[i])+'\t')
    for j in range(len(new_trainTs[i])):
        dataset.write(str(new_trainTs[i][j])+'\t')
    dataset.write('\n')
dataset.close()

#embeddings
if dynamic_traning is True:
    embeddings=tf.Variable(tf.random_uniform([len(voc_poi),embedding_size],-1.0,1.0))
    time_embeddings = tf.Variable(tf.random_uniform([24,32], -1.0, 1.0))
    distance_embeddings1=tf.Variable(tf.random_uniform([32], -1.0, 1.0))
    distance_embeddings2=tf.Variable(tf.random_uniform([32], -1.0, 1.0))
    weights=tf.Variable(tf.truncated_normal([embedding_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
    bias=tf.Variable(tf.zeros([embedding_size]),dtype=tf.float32)
    embeddings=tf.nn.xw_plus_b(embeddings,weights,bias)
else:
    embeddings=[]
    fread_emb=open('data/'+embedding_name+'vec.dat','r')
    for line in fread_emb.readlines():
        lineArr=line.split()
        temp=list()
        for i in range(1,len(lineArr)):
            temp.append(float(lineArr[i]))
        embeddings.append(temp)
    embeddings=tf.constant(embeddings)

# =============================== data load end====================================== #

# =============================== tf.vars ====================================== #
keep_prob = tf.placeholder("float")
lens=tf.placeholder(dtype=tf.int32)
input_X=tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
input_X_de=tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
input_t=tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
input_d1=tf.placeholder(dtype=tf.float32, shape=[batch_size, None])
input_d2=tf.placeholder(dtype=tf.float32, shape=[batch_size, None])
target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')

code = tf.placeholder(tf.float32, shape=[None, c_dim])
z = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
z_t=tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
z_d1=tf.placeholder(dtype=tf.float32, shape=[batch_size, None])
z_d2=tf.placeholder(dtype=tf.float32, shape=[batch_size, None])

# =============================== Encoder ====================================== #
def encoder(X,context,keep_prob=0.5):
    """
    encode discrete feature to continuous latent vector
    :param tensor: [batch_size,length,embedding_size].
    :return:encoded latent vector
    """
    with tf.variable_scope("encoder"):
        tensor=tf.nn.embedding_lookup(embeddings,X) #find embeddings of trajectory:[batch_size,length,embedding_size].
        time_t=tf.nn.embedding_lookup(time_embeddings,context[0])
        print 'time_t',time_t
        space=tf.tensordot(context[1],distance_embeddings1,0)+tf.tensordot(context[2],distance_embeddings2,0)
        print 'space',space
        tensor=tf.concat([tensor,time_t],2)
        tensor=tf.concat([tensor,space],2)
        trans_tensor=tf.transpose(tensor,[1,0,2])       #[length,batch_size,embedding_size].
        lstm_cell=tf.nn.rnn_cell.LSTMCell(n_hidden)
        dr_lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=keep_prob)
        (output,states)=tf.nn.dynamic_rnn(dr_lstm_cell,trans_tensor,time_major=True,dtype=tf.float32)
        latent_code=output[-1]
        latent_code=tf.nn.l2_normalize(latent_code)
        print 'latentcode',latent_code
        #latent_code= fully_connected(latent_code, c_dim, initializer=initializer, is_last=True, scope="encoder_output")
        return latent_code,states

# =============================== Decoder ====================================== #
def decoder(tensor,X,en_state,reuse=False):
    """
     decode continuous vector to probability of pixel
     :param tensor: 2-D tensor.
     :param output_dim: Integer. dimension of output
     :param is_mnist: Boolean. mnist or not
     :param reuse: Boolean. reuse or not
     :param is_train:Boolean. train or not
     :return: 2-D tensor. decoded vector (image)
     """
    with tf.variable_scope('decoder',reuse=reuse) as scope:

        decode_lstm=tf.nn.rnn_cell.LSTMCell(n_hidden)
        decode_dr_lstm = tf.nn.rnn_cell.DropoutWrapper(decode_lstm, output_keep_prob=keep_prob)
        output_layer=Dense(len(vocab_to_int))
        decoder_initial_state=en_state#LSTMStateTuple(c_state, h_state)

        copy = tf.tile(tf.constant([vocab_to_int['GO']]), [batch_size])
        training_helper = seqc.GreedyEmbeddingHelper2(embeddings,
                                                         sequence_length=target_sequence_length, start_tokens=copy)
        training_decoder = seqc.BasicDecoder(decode_dr_lstm, training_helper, decoder_initial_state,tensor,output_layer)  # cell,helper, initial_state, out_layer(convert rnn_size to vocab_size)
        output, _, _ = seqc.dynamic_decode(training_decoder,
                                              impute_finished=True,
                                              maximum_iterations=max_target_sequence_length)
        predicting_logits = tf.identity(output.sample_id, name='predictions')
        training_logits = tf.identity(output.rnn_output, 'logits')
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        target = X
        return output, predicting_logits, training_logits, masks, target


# =============================== Generator ====================================== #
def generator(z,context,reuse=False):
    """
    generator of WGAN
    :param z: 2-D tensor. noise with standard normal distribution
    :param reuse: Boolean. reuse or not
    :return: 2-D tensor. latent vector
    """
    """
    encode discrete feature to continuous latent vector
    :param tensor: [batch_size,length,embedding_size].
    :return:encoded latent vector
    """
    with tf.variable_scope("generator"):
        tensor=tf.nn.embedding_lookup(embeddings,z) #find embeddings of trajectory:[batch_size,length,embedding_size].
        time_t=tf.nn.embedding_lookup(time_embeddings,context[0])
        space=tf.tensordot(context[1],distance_embeddings1,0)+tf.tensordot(context[2],distance_embeddings2,0)
        tensor=tf.concat([tensor,time_t],2)
        tensor=tf.concat([tensor,space],2)
        trans_tensor=tf.transpose(tensor,[1,0,2])       #[length,batch_size,embedding_size].
        lstm_cell=tf.nn.rnn_cell.LSTMCell(n_hidden)
        dr_lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=keep_prob)
        (output,states)=tf.nn.dynamic_rnn(dr_lstm_cell,trans_tensor,time_major=True,dtype=tf.float32)
        latent_code=output[-1]
        latent_code=tf.nn.l2_normalize(latent_code)
        return latent_code,states
# =============================== Discriminator(Critic) ====================================== #
def critic(latent,reuse=False):
    """
    discriminator of WGAN
    :param latent: 2-D tensor. latent vector
    :param reuse: Boolean. reuse or not
    :return: 2-D tensor. logit of data or noise
    """
    with tf.variable_scope("critic",reuse=reuse):
        fc_100 = fully_connected(latent, 100, initializer=initializer, scope="fc_100")
        fc_60 = fully_connected(fc_100, 60, initializer=initializer, scope="fc_60")
        fc_20 = fully_connected(fc_60, 20, initializer=initializer, scope="fc_20")
        output=fully_connected(fc_20,1,initializer=initializer,is_last=True,scope="critic_output")
        #WGAN does not using activate
    return  output

# =============================== Function ====================================== #
def autoencoder(X,de_X,context,keep_prob):
    """
    deep autoencoder. reconstruction the input data
    :param data: 2-D tensor. data for reconstruction
    :return: 2-D tensor. reconstructed data and latent vector
    """
    with tf.variable_scope("autoencoder"):
        latent,en_state=encoder(X,context,keep_prob)
        output_, predicting_logits_, training_logits_, masks_, target_=decoder(latent,de_X,en_state)
    return training_logits_,masks_,target_,latent,predicting_logits_,en_state
def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])  # 取最大长度
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def pad_time_batch(time_batch):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in time_batch])  # 取最大长度
    return [sentence + [0] * (max_sentence - len(sentence)) for sentence in time_batch]
def pad_dist_batch(dist_batch):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in dist_batch])  # 取最大长度
    return [sentence + [sentence[-1]] * (max_sentence - len(sentence)) for sentence in dist_batch]
def eos_sentence_batch(sentence_batch, eos_in):
    return [sentence + [eos_in] for sentence in sentence_batch]

# =============================== Graph ====================================== #
"""
build network
:return:
"""
context=[input_t,input_d1,input_d2]
z_context=[z_t,z_d1,z_d2]
training_logits_,masks_,target_,real_code,predicting_logits_,encoder_state=autoencoder(input_X,input_X_de,context,keep_prob)
g_code,g_state=generator(z,z_context) #flat c
print g_code
critic_real=critic(real_code)
print critic_real
critic_fake=critic(g_code,reuse=True)
print critic_fake
#WGAN loss
disc_real_loss=tf.reduce_mean(critic_real)
disc_fake_loss=-tf.reduce_mean(critic_fake) #Negative sample phase
critic_loss=tf.reduce_mean(critic_real)-tf.reduce_mean(critic_fake)
gen_loss=tf.reduce_mean(critic_fake) #Train the generator

#for discrete input, use cross entropy loss
AE_loss=seqc.sequence_loss(training_logits_, target_, masks_)

#get trainable variables
AE_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="autoencoder")
gen_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="generator")
critic_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='critic')
encoder_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='autoencoder/encoder')

#set optimizer for each module
disc_op=tf.train.AdamOptimizer(learning_rate=critic_lr)
gen_op=tf.train.AdamOptimizer(learning_rate=gen_lr)
AE_op=tf.train.GradientDescentOptimizer(learning_rate=AE_learning_rate)


#compute gradients
pos_critic_grad=disc_op.compute_gradients(disc_real_loss,critic_variables+encoder_variables)

neg_critic_grad=disc_op.compute_gradients(disc_fake_loss,critic_variables)

#clipping gradients for negative samples
neg_critic_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in neg_critic_grad]
gen_grad = gen_op.compute_gradients(gen_loss, gen_variables)
AE_grad = AE_op.compute_gradients(AE_loss,AE_variables)
#AE_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in AE_grad]

#apply gradients
update_critic_pos=disc_op.apply_gradients(pos_critic_grad)
update_critic_neg=disc_op.apply_gradients(neg_critic_grad)
update_G=gen_op.apply_gradients(gen_grad)
update_AE=AE_op.apply_gradients(AE_grad)

#reconstruction
with tf.variable_scope("autoencoder"):
    toutput_, tpredicting_logits_, ttraining_logits_, tmasks_, ttarget_=decoder(real_code,input_X_de,encoder_state,reuse=True)
    foutput_, fpredicting_logits_, ftraining_logits_, fmasks_, ftarget_ = decoder(g_code, input_X_de,g_state,reuse=True)
def train(data):
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        train_variables, test_variables=data
        encoder_train, decoder_train, train_batch_lenth, n_trainTime, n_trainDist1, n_trainDist2, z_train, z_train_time, z_train_dist1, z_train_dist2=train_variables

        max_F1=[]
        max_pair_F1=[]
        test_max_F1=[]
        test_max_pair_F1=[]
        ftest_max_F1=[]
        ftest_max_pair_F1=[]
        res = {}
        fres={}
        for epoch in range(train_iters):
            step = 0
            ACC=0
            F1=[]
            pairs_F1=[]
            LOSS=[]
            gen_LOSS=[]
            Critic_Loss=[]
            Fake_Loss=[]
            GAN_Loss=[]
            while step < len(encoder_train) // batch_size:
                start_i = step * batch_size
                dist_1_=n_trainDist1[start_i:start_i + batch_size]
                dist_2_=n_trainDist2[start_i:start_i + batch_size]
                input_time_=n_trainTime[start_i:start_i + batch_size]
                encode_batch =encoder_train[start_i:start_i + batch_size]
                decode_batch = decoder_train[start_i:start_i + batch_size]
                pad_source_lengths =train_batch_lenth[start_i:start_i + batch_size]
                z_in =z_train[start_i:start_i + batch_size]
                z_time=z_train_time[start_i:start_i + batch_size]
                z_dist1=z_train_dist1[start_i:start_i + batch_size]
                z_dist2=z_train_dist2[start_i:start_i + batch_size]

                #update AE
                _,_AEloss=sess.run([update_AE,AE_loss],feed_dict={target_sequence_length: pad_source_lengths,
                           input_X: encode_batch,input_X_de: decode_batch,input_t:input_time_,input_d1:dist_1_,input_d2:dist_2_,keep_prob: 0.5})
                # update critic & encoder at positive sample phase
                for k in range(10):
                    _, _critic_loss, real_loss = sess.run([update_critic_pos,critic_loss,disc_real_loss],feed_dict={target_sequence_length: pad_source_lengths,
                           input_X: encode_batch,input_X_de: decode_batch,z: z_in,z_t:z_time,z_d1:z_dist1,z_d2:z_dist2,input_t:input_time_,input_d1:dist_1_,input_d2:dist_2_,keep_prob: 0.5})
                    fake_loss, _ = sess.run([disc_fake_loss, update_critic_neg], feed_dict={z: z_in, z_t:z_time,z_d1:z_dist1,z_d2:z_dist2,keep_prob: 0.5})
                # update generate
                _, gan_loss = sess.run([update_G, gen_loss], feed_dict={z: z_in, z_t:z_time,z_d1:z_dist1,z_d2:z_dist2,keep_prob: 0.5})
                #training result
                values= sess.run(predicting_logits_, feed_dict={target_sequence_length: pad_source_lengths,
                                                                   input_X: encode_batch,input_X_de: decode_batch,input_t:input_time_,input_d1:dist_1_,input_d2:dist_2_,
                                                                z_t: z_time, z_d1: z_dist1, z_d2: z_dist2,keep_prob: 0.5})
                LOSS.append(_AEloss)
                gen_LOSS.append(real_loss)
                Critic_Loss.append(_critic_loss)
                Fake_Loss.append(fake_loss)
                GAN_Loss.append(gan_loss)

                for v in range(len(values)):
                    length=pad_source_lengths[v]-1
                    if (decode_batch[v][:length]==values[v][:length]).all():
                        ACC+=1
                    f=calc_F1(decode_batch[v][:length],values[v][:length])
                    p_f=calc_pairsF1(decode_batch[v][:length],values[v][:length])
                    F1.append(f)
                    pairs_F1.append(p_f)
                step += 1
            #saver.save(sess, model)
            #print 'train F1',np.mean(F1)
            #print 'pairs-F1',np.mean(pairs_F1)
            #print 'epoch',epoch,'train accuracy',ACC/len(trainU)
            #print 'testing------------------>loss',epoch,np.sum(LOSS)
            test_f1,test_pairs_f1,ftest_f1,ftest_pairs_f1=test(sess=sess,test_variables=test_variables) #
            max_F1.append(np.mean(F1))
            max_pair_F1.append(np.mean(pairs_F1))
            #print test_f1,test_pairs_f1,ftest_f1,ftest_pairs_f1
            max_epoch = -1
            res.setdefault(test_f1, []).append(test_pairs_f1)
            fres.setdefault(ftest_f1, []).append(ftest_pairs_f1)
        keys = res.keys()
        fkeys =fres.keys()
        keys = sorted(keys)
        fkeys = sorted(fkeys)
        return (max_F1[max_epoch]), (max_pair_F1)[max_epoch],keys[-1],max(res[keys[-1]]),fkeys[-1],max(fres[fkeys[-1]])

def test(sess,test_variables):#,Test_Time=None
    step = 0
    Pred_ACC = 0
    F1 = []
    pairs_F1 = []
    fake_F1=[]
    fake_pairs_F1=[]
    encoder_test, decoder_test, test_batch_lenth, n_testTime, n_testDist1, n_testDist2, z_test, z_test_time, z_test_dist1, z_test_dist2 = test_variables
    while step < len(encoder_test) // batch_size:
        start_i = step * batch_size
        dist_1_ = n_testDist1[start_i:start_i + batch_size]
        dist_2_ = n_testDist2[start_i:start_i + batch_size]
        input_time_ = n_testTime[start_i:start_i + batch_size]
        encode_batch = encoder_test[start_i:start_i + batch_size]
        decode_batch = decoder_test[start_i:start_i + batch_size]
        pad_source_lengths = test_batch_lenth[start_i:start_i + batch_size]
        z_in = z_test[start_i:start_i + batch_size]
        z_time = z_test_time[start_i:start_i + batch_size]
        z_dist1 = z_test_dist1[start_i:start_i + batch_size]
        z_dist2 = z_test_dist2[start_i:start_i + batch_size]
        otpredicting_logits_,fake_ = sess.run([tpredicting_logits_,fpredicting_logits_],feed_dict={target_sequence_length: pad_source_lengths,
                                                   input_X: z_in,input_t:z_time,input_d1:z_dist1,input_d2:z_dist2,z:z_in,z_t:z_time,z_d1:z_dist1,z_d2:z_dist2,
                                                   keep_prob: 1.0})  # 梯度更新后的结果

        for v in range(len(otpredicting_logits_[:1])):
            length = pad_source_lengths[v] - 1
            if (decode_batch[v][:length]== otpredicting_logits_[v][:length]).all():
                Pred_ACC += 1
            actual = decode_batch[v][:length]
            recommend = np.concatenate([[actual[0]], otpredicting_logits_[v][1:length - 1]], axis=0)
            recommend = np.concatenate([recommend, [actual[-1]]], axis=0)

            frecommend = np.concatenate([[actual[0]], fake_[v][1:length - 1]], axis=0)
            frecommend = np.concatenate([frecommend, [actual[-1]]], axis=0)
            #print actual,recommend,frecommend
            f_f = calc_F1(actual, frecommend)
            f_p_f = calc_pairsF1(actual, frecommend)
            f = calc_F1(actual, recommend)
            p_f = calc_pairsF1(actual, recommend)
            F1.append(f)
            pairs_F1.append(p_f)
            fake_F1.append(f_f)
            fake_pairs_F1.append(f_p_f)
        step += 1
    #print 'trajectory length',length
    return np.mean(F1),np.mean(pairs_F1),np.mean(fake_F1),np.mean((fake_pairs_F1))
def get_data(index,K):
    # sort original data
    index_T = {}
    trainT = []
    trainU = []
    trainTime=[]
    trainDist=[]
    for i in range(len(new_trainTs)):
        index_T[i] = len(new_trainTs[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])
    for i in range(len(temp_size)):
        id = temp_size[i][0]
        trainT.append(new_trainTs[id])
        trainU.append(TRAIN_USER[id])
        trainTime.append(TRAIN_TIME[id])
        trainDist.append(TRAIN_DIST[id])
    value=int(math.ceil(len(trainT)/K))
    if index==K-1:
        testT=trainT[-value:]
        testU=trainU[-value:]
        trainT=trainT[:-value]
        trainU=trainU[:-value]

        testTime=trainTime[-value:]
        testDist=trainDist[-value:]
        trainTime=trainTime[:-value]
        trainDist=trainDist[:-value]

    elif index==0:
        testT=trainT[:(index+1)*value]
        testU=trainU[:(index+1)*value]
        trainT =trainT[(index+1)*value:]
        trainU =trainU[(index+1)*value:]

        testTime=trainTime[:(index+1)*value]
        testDist=trainDist[:(index+1)*value]
        trainTime=trainTime[(index+1)*value:]
        trainDist=trainDist[(index+1)*value:]

    else:
        testT=trainT[index*value:(index+1)*value]
        testU=trainU[index*value:(index+1)*value]
        trainT = trainT[0:index*value]+trainT[(index+1)*value:]
        trainU = trainU[0:index*value]+trainU[(index+1)*value:]

        testTime=trainTime[index*value:(index+1)*value]
        testDist=trainDist[index*value:(index+1)*value]
        trainTime=trainTime[0:index*value]+trainTime[(index+1)*value:]
        trainDist=trainDist[0:index*value]+trainDist[(index+1)*value:]
    train_size = len(trainT) % batch_size
    #if
    trainT = trainT + [trainT[-1]]*(batch_size-train_size)  # copy data and fill the last batch size
    trainU = trainU + [trainU[-1]]*(batch_size-train_size)
    trainTime=trainTime+[trainTime[-1]]*(batch_size-train_size)
    trainDist = trainDist + [trainDist[-1]] * (batch_size - train_size)
    #print 'Text', testT,index,K
    test_size = len(testT) % batch_size
    if test_size!=0:
        testT = testT + [testT[-1]]*(batch_size-test_size)  # copy data and fill the last batch size
        testU = testU + [testU[-1]]*(batch_size-test_size)  #BUG for test_size<batch_size len(train_size<test_size)
        testTime=testTime+[testTime[-1]]*(batch_size-test_size)
        testDist = testDist + [testDist[-1]] * (batch_size - test_size)
    print 'test size',test_size,len(testT)
    #pre-processing
    step=0
    encoder_train=[]
    decoder_trian=[]
    encoder_test=[]
    decoder_test=[]
    n_trainTime=[]
    n_testTime=[]
    n_trainDist1=[]
    n_trainDist2= []
    n_testDist1=[]
    n_testDist2= []
    train_batch_lenth=[]
    test_batch_lenth=[]
    z_train=[]
    z_train_time=[]
    z_train_dist1=[]
    z_train_dist2 = []
    z_test=[]
    z_test_time=[]
    z_test_dist1=[]
    z_test_dist2 =[]
    while step < len(trainU) // batch_size:
        start_i = step * batch_size
        input_x = trainT[start_i:start_i + batch_size]
        #time
        input_time = trainTime[start_i:start_i + batch_size]
        input_time_ = pad_time_batch(input_time)
        input_d = trainDist[start_i:start_i + batch_size]
        #input
        encode_batch = pad_sentence_batch(input_x, vocab_to_int['PAD'])
        decode_batchs = []
        z_batch=[]
        z_batch_time=[]
        z_batch_dist1=[]
        z_batch_dist2=[]
        for sampe in input_x:
            value = sampe
            value_=[sampe[0],sampe[-1]]
            decode_batchs.append(value)
            z_batch.append(value_)
        for sample in input_time:
            z_batch_time.append([sample[0],sample[-1]])
        decode_batch_ = eos_sentence_batch(decode_batchs, vocab_to_int['END'])
        decode_batch = pad_sentence_batch(decode_batch_, vocab_to_int['PAD'])

        dist_1 = []
        dist_2 = []
        # print 'value',input_d
        for i in range(len(input_d)):
            temp_dist1 = []
            temp_dist2 = []
            for j in range(len(input_d[i])):
                temp_dist1.append(input_d[i][j][0])
                temp_dist2.append(input_d[i][j][1])
            dist_1.append(temp_dist1)
            dist_2.append(temp_dist2)
            z_batch_dist1.append([temp_dist1[0],temp_dist1[-1]])
            z_batch_dist2.append([temp_dist2[0], temp_dist2[-1]])
        dist_1_ = pad_dist_batch(dist_1)
        dist_2_ = pad_dist_batch((dist_2))

        pad_source_lengths = []
        for source in decode_batchs:
            pad_source_lengths.append(len(source) + 1)
        for i in range(batch_size):
            encoder_train.append(encode_batch[i])
            decoder_trian.append(decode_batch[i])
            train_batch_lenth.append(pad_source_lengths[i])
            n_trainTime.append(input_time_[i])
            n_trainDist1.append(dist_1_[i])
            n_trainDist2.append(dist_2_[i])
            z_train.append(z_batch[i])
            z_train_time.append(z_batch_time[i])
            z_train_dist1.append(z_batch_dist1[i])
            z_train_dist2.append(z_batch_dist2[i])
        step+=1
        #append to
    steps=0
    while steps < len(testU) // batch_size:
        start_i = steps * batch_size
        input_x = testT[start_i:start_i + batch_size]
        # time
        input_time = testTime[start_i:start_i + batch_size]
        input_time_ = pad_time_batch(input_time)
        input_d = testDist[start_i:start_i + batch_size]
        # input
        encode_batch = pad_sentence_batch(input_x, vocab_to_int['PAD'])
        decode_batchs = []
        z_batch = []
        z_batch_time = []
        z_batch_dist1 = []
        z_batch_dist2 = []
        for sampe in input_x:
            value = sampe
            value_ = [sampe[0], sampe[-1]]
            decode_batchs.append(value)
            z_batch.append(value_)
        for sample in input_time:
            z_batch_time.append([sample[0], sample[-1]])
        decode_batch_ = eos_sentence_batch(decode_batchs, vocab_to_int['END'])
        decode_batch = pad_sentence_batch(decode_batch_, vocab_to_int['PAD'])

        dist_1 = []
        dist_2 = []
        # print 'value',input_d
        for i in range(len(input_d)):
            temp_dist1 = []
            temp_dist2 = []
            for j in range(len(input_d[i])):
                temp_dist1.append(input_d[i][j][0])
                temp_dist2.append(input_d[i][j][1])
            dist_1.append(temp_dist1)
            dist_2.append(temp_dist2)
            z_batch_dist1.append([temp_dist1[0], temp_dist1[-1]])
            z_batch_dist2.append([temp_dist2[0], temp_dist2[-1]])
        dist_1_ = pad_dist_batch(dist_1)
        dist_2_ = pad_dist_batch((dist_2))

        pad_source_lengths = []
        for source in decode_batchs:
            pad_source_lengths.append(len(source) + 1)

        for i in range(batch_size):
            encoder_test.append(encode_batch[i])
            decoder_test.append(decode_batch[i])
            test_batch_lenth.append(pad_source_lengths[i])
            n_testTime.append(input_time_[i])
            n_testDist1.append(dist_1_)
            n_testDist2.append(dist_2_)
            z_test.append(z_batch[i])
            z_test_time.append(z_batch_time[i])
            z_test_dist1.append(z_batch_dist1[i])
            z_test_dist2.append(z_batch_dist2[i])
        steps+=1
    train_variables=[encoder_train,decoder_trian,train_batch_lenth, n_trainTime,n_trainDist1, n_trainDist2,z_train,z_train_time,z_train_dist1,z_train_dist2]
    test_variables = [encoder_test, decoder_test, test_batch_lenth, n_testTime, n_testDist1, n_testDist2,
                       z_test, z_test_time, z_test_dist1, z_test_dist2]
    return train_variables,test_variables
# main--
if __name__ == "__main__":
    K=len(TRAIN_TRA)#1fold
    print 'K',K
    Tr_F1=[]
    Tr_pairsF1=[]
    Te_F1=[]
    Te_pairsF1=[]
    fTe_F1=[]
    fTe_pairsF1=[]
    for i in range(K):
        data=get_data(index=i,K=K)
        train_F1,train_pairs_F1,test_F1,test_pairs_F1,ftest_F1,ftest_pairs_F1=train(data)
        Tr_F1.append(train_F1)
        Tr_pairsF1.append(train_pairs_F1)
        Te_F1.append(test_F1)
        Te_pairsF1.append(test_pairs_F1)
        fTe_F1.append(ftest_F1)
        fTe_pairsF1.append(ftest_pairs_F1)
        print 'K=', i,train_F1,train_pairs_F1,test_F1,test_pairs_F1,ftest_F1,ftest_pairs_F1
    print 'model output Train F1,',np.mean(Tr_F1),'Train pairs F1',np.mean(Tr_pairsF1),'Test F1',np.mean(Te_F1),'Test pairs F1',np.mean(Te_pairsF1)
    print 'model output std,', np.std(Te_F1), np.std(Te_pairsF1)
    print 'model test',np.mean(fTe_F1),np.mean(fTe_pairsF1),np.std(fTe_F1),np.std(fTe_pairsF1)
    real=open('./result/gae'+tra_name+'.dat','w')
    real.write('data name'+str(tra_name)+'\n')
    real.write('Encoder'+'\tF1\t'+str(np.mean(Te_F1))+'\tstd\t'+str(np.std(Te_F1))+'\t pairs-F1\t'+str(np.mean(Te_pairsF1))+'\tstd\t'+str(np.std(Te_pairsF1))+'\n')
    real.write(('GANs'+'\tF1\t'+str(np.mean(fTe_F1))+'\tstd\t'+str(np.std(fTe_F1))+'\t pairs-F1\t'+str(np.mean(fTe_pairsF1))+'\tstd\t'+str(np.std(fTe_pairsF1))+'\n'))