import re
from openai import OpenAI
import textwrap
import os
import torch
import numpy as np
import argparse
from os.path import join as pjoin
import utils1.paramUtil as paramUtil
from torch.utils.data import DataLoader
from utils1.plot_script import *
from utils1.get_opt import get_opt
from datasets.evaluator_models import MotionLenEstimatorBiGRU
from trainers import DDPMTrainer
from models import MotionTransformer
from utils1.word_vectorizer import WordVectorizer, POS_enumerator
from utils1.utils import *
from utils1.motion_process import recover_from_ric


def plot_t2m(data, result_path, npy_path, caption):
    joint = recover_from_ric(torch.from_numpy(data).float(), opt.joints_num).numpy()
    joint = motion_temporal_filter(joint, sigma=1)
    plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
    if npy_path != "":
        np.save(npy_path, joint)


def build_models(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder



client = OpenAI()
prompt_system = """
You are a human who acts in accordance to social norms that govern how humans behave around other humans in a given context.
Rules:
-[Scenario] describes the scenario in which you find yourself.
- Generate [What human would do] and [What human would say] which describes what you would do in the given [Scenario].
- Before producing [What human would do] and [What human would say], describe you [Reasoning] about how to act given the [Scenario].
- Generate [Reasoning] inside [Start reasoning] and [End reasoning] in point-form
- Generate [What human would do] inside [Start what human would do] and [End what human would do] in point-form
- [What human would do] describes people's body movement feedback in response to this input.
- Generate [What human would say] inside [Start what human would say] and [End what human would say] in point-form
- Do not leave ambiguity in the behaviors, Respond according to norms.
"""

prompt_example = """
[Scenario]
A person just greeted you by say [hello].
[Start reasoning]
- The person has initiated a social interaction by greeting me.
- It is a common social norm to respond when someone greets you.
- Ignoring the greeting could be seen as rude or dismissive.
[End reasoning]
[Start what human would say]
- hello    
[End what human would say]
[Start what human would do]
- smile to show friendliness and openness
- wave hands to greet him    
[End what human would do]
"""
prompt_example1="""
[Scenario]
- Another person asked you the location of the tea cup.
[Start reasoning]
- They must be looking for the tea cup, probably to use it.
- They believe I might know where it is.\n- It\'s customary to assist others when they're looking for something and you know where it is.
- Ignoring the request or refusing to help without good reason would be considered rude.
[End reasoning]
[Start what human would do]
- Recall the last place I've seen the cup or where it's usually kept.
- Direct them to the location.
[End what human would do]
[Start what human would say]
- The tea cup is in the kitchen cabinet next to the stove, 
[End what human would say]
""" 

prompt_example2="""
[Scenario]
A person wants you to tell a story about a dragon.
[Start reasoning]
- They\'re requesting a specific kind of story, so they probably have an interest in dragons or fantasy stories.
- It's polite to respond to requests for stories by telling one if you are able, as it fosters connection and conversation. 
- As the request is specific, it\'s important to make the dragon a central part of the story to satisfy their request.
[End reasoning]
[Start what human would do]
- do tha action to pretend to think.
- pretending to have a flash of inspiration and think of a suitable story.
- while telling the story,use upper body movements to show what the dragon looks like.
[End what human would do]
[Start what human would say]
- "Once upon a time, in a kingdom far far away, there lived a fearsome dragon with gleaming scales of emerald green...."  
[End what human would say]
"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, default="checkpoints/t2m/t2m_motiondiffuse/opt.txt", help='Opt path')
    parser.add_argument('--text', type=str, default="a person is drinking the tea.", help='Text description for motion generation')
    parser.add_argument('--motion_length', type=int, default=198, help='Number of frames for motion generation')
    parser.add_argument('--result_path', type=str, default="test_sample.gif", help='Path to save generation result')
    parser.add_argument('--npy_path', type=str, default="test_sample.npy", help='Path to save 3D keypoints sequence')
    parser.add_argument('--gpu_id', type=int, default=0, help="which gpu to use")
    args = parser.parse_args()

    # prompt_system = textwrap.dedent(prompt_system)
    # prompt_system = prompt_system.split('\n', 1)[1]

    # prompt_example = textwrap.dedent(prompt_example)
    # prompt_example = prompt_example.split('\n', 1)[1]

    # prompt_example1 = textwrap.dedent(prompt_example1)
    # prompt_example1 = prompt_example1.split('\n', 1)[1]

    # response = client.chat.completions.create(
    #   model="gpt-4-0613",
    #   messages=[
    #     {"role": "system", "content": prompt_system},
    #     {"role": "user", "content": "hello!"},
    #     {"role": "assistant", "content": prompt_example},
    #     {"role": "user", "content": "where is the tea cup?."},
    #     {"role": "assistant", "content": prompt_example1},
    #     {"role": "user", "content": "Please tell me a story about dragon."},
    #     {"role": "assistant", "content": prompt_example2},
    #     {"role": "user", "content":"Please pretend to take pictures with your mobile phone."},
    #   ]
    # )
    # text=str(response.choices[0].message)
    # print(text)
    # print(text)
    # pattern = r'\[Start what human would do\](.*?)\[End what human would do\]'
    # result = re.search(pattern, text, re.DOTALL)

    # if result:
    #     extracted_text = result.group(1).strip()
    #     print(extracted_text)
    # else:
    #     print("未找到匹配的文本")

    # cleaned_text = extracted_text.replace("\n", "")

    # # print(args.text)
    # # args.text = cleaned_text
    # print(cleaned_text)

    # args.text = "Pick up the imaginary mobile phone. Position the phone in landscape or portrait mode in front of me. Focus the middle finger on the screen as if focusing the camera. Pretend to press the shutter button on the phone screen."
    # args.text = "Pretend to hold a guitar and do the movement typical of a metal guitar player, moving head rhythmically, as if headbanging."
    # args.text = " Position hands as if holding drumsticks. Mimic common drumming movements, like striking a snare drum or cymbal. Coordinate body rhythm as if playing along with a song. "
    
    args.text = "Smile and wave hand."
    device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')
    opt = get_opt(args.opt_path, device)
    opt.do_denoise = True

    assert opt.dataset_name == "t2m"
    # assert opt.dataset_name == "kit"
    
    assert args.motion_length <= 3600
    opt.data_root = './dataset/HumanML3D'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.joints_num = 22
    opt.dim_pose = 263
    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    num_classes = 200 // opt.unit_length

    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    encoder = build_models(opt).to(device)
    trainer = DDPMTrainer(opt, encoder)
    trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

    trainer.eval_mode()
    trainer.to(opt.device)

    result_dict = {}
    with torch.no_grad():
        if args.motion_length != -1:
            caption = [args.text]
            m_lens = torch.LongTensor([args.motion_length]).to(device)
            pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
            motion = pred_motions[0].cpu().numpy()
            motion = motion * std + mean
            print(motion)
            title = args.text + " #%d" % motion.shape[0]
            plot_t2m(motion, args.result_path, args.npy_path, title)


# %%%此处我们将讲生成的机器人动作映射到机器人上；


# prompt_system = textwrap.dedent(prompt_system)
# prompt_system = prompt_system.split('\n', 1)[1]

# prompt_example = textwrap.dedent(prompt_example)
# prompt_example = prompt_example.split('\n', 1)[1]

# prompt_example1 = textwrap.dedent(prompt_example1)
# prompt_example1 = prompt_example1.split('\n', 1)[1]

# response = client.chat.completions.create(
#   model="gpt-4-0613",
#   messages=[
#     {"role": "system", "content": prompt_system},
#     {"role": "user", "content": "hello!"},
#     {"role": "assistant", "content": prompt_example},
#     {"role": "user", "content": "where is the tea cup?."},
#     {"role": "assistant", "content": prompt_example1},
#     {"role": "user", "content": "Please tell me a story about dragon."},
#     {"role": "assistant", "content": prompt_example2},
#     {"role": "user", "content": "See you!"},
#   ]
# )
# print(response.choices[0].message)

# ChatCompletionMessage(content='[Scenario]
# A person asks you to pretend to play the drums.
# [Start reasoning]
#  The person is requesting a playful and fun action.
#  Social norms suggest that it\'s friendly and enjoyable to participate in such requests, assuming it\'s done in a casual and appropriate setting.
#  It\'s particularly important to remain respectful of the surroundings and not make too much noise or cause disruption.
# [End reasoning]
# [Start what human would do]
#  Position hands as if holding drumsticks.
#  Mimic common drumming movements, like striking a snare drum or cymbal. 
#  Coordinate body rhythm as if playing along with a song. 
# Tap feet lightly, mimicking the use of a bass drum pedal.
# [End what human would do]
# [Start what human would say]
#  "Sure, here we go! Boom, Boom, Tsh, Boom, Boom, Tsh!" (mimicking the sound of the drums while playing)
# [End what human would say]', 
# role='assistant', function_call=None, tool_calls=None)

# ChatCompletionMessage(content='[
# Scenario]
# A person is saying goodbye to you with the phrase, "See you!"
# [Start reasoning]
#  The individual is ending a conversation or leaving, and they are using common parting words.
#  It\'s polite to give a response when someone says goodbye.
#  Just ignoring the farewell might be regarded as impolite or dismissive.
# [End reasoning]
# [Start what human would do]
#  smile and wave hand.
# [End what human would do]\n- [Start what human would say]
#  "See you later!"
# [End what human would say]', role='assistant', function_call=None, tool_calls=None)