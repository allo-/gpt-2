#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
try:
    import readline
except:
    pass

import model, sample, encoder_sp as encoder

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    run_name='run1',
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name, 'checkpoint/%s' % run_name))
        saver.restore(sess, ckpt)
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        print("The model has", total_parameters, "parameters")


        multi_line_input = False
        while True:
            if multi_line_input:
                raw_text = ""
                while not raw_text:
                    line = input('Model prompt (end the prompt with "END") >>> ')
                    while line != "END":
                        raw_text += line + "\n"
                        line = input(">>> ")
                        if line.startswith("!"):
                            break
                    if not raw_text:
                        print('Prompt should not be empty!')
                raw_text = raw_text.strip()
                raw_text = raw_text.replace("NEWLINE", "") # allow to add new lines the end as NL
            else:
                raw_text = input("Model prompt >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("Model prompt >>> ")

            try:
                if raw_text.startswith("!length"):
                    _, new_length = raw_text.split(" ")
                    length = int(new_length)
                if raw_text.startswith("!temp"):
                    _, new_temperature = raw_text.split(" ")
                    temperature = float(new_temperature)
                if raw_text.startswith("!top_k"):
                    _, new_top_k = raw_text.split(" ")
                    top_k = int(new_top_k)
                if raw_text.startswith("!nsamples"):
                    _, nsamples = raw_text.split(" ")
                    nsamples = int(nsamples)
                if raw_text.startswith("!multiline"):
                    multi_line_input = not multi_line_input
            except ValueError:
                print("Invalid value")
                continue

            if raw_text.startswith("!"):
                print("Changing model parameters: length={}, "
                "temperature={}, top_k={}, nsamples={}, "
                "multiline={}".format(length, temperature, top_k,
                    nsamples, multi_line_input))

                output = sample.sample_sequence(
                    hparams=hparams, length=length,
                    context=context,
                    batch_size=batch_size,
                    temperature=temperature, top_k=top_k
                )
                continue

            print('')
            print('')
            print('Generating samples ...')
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    text = "\n".join([line.strip() for line in text.split("\n")])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
                    print()
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)

