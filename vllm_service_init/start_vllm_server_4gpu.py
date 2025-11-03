#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
'''  
4 GPU version of the vLLM evaluation server with tensor parallelism.  
This version uses all 4 GPUs via tensor parallelism for a single model instance.  
  
Usage:  
    python start_vllm_server_4gpu.py --port 5000 --model_path Qwen/Qwen3-4B-Base --tensor_parallel_size 4  
'''  
  
from flask import Flask, request, jsonify  
import vllm  
import argparse  
import json  
import os  
import threading  
import time  
import torch  
from transformers import AutoTokenizer  
from mathruler.grader import extract_boxed_content, grade_answer  
import stopit  
  
# ------------------------- Command-Line Arguments ------------------------- #  
parser = argparse.ArgumentParser()  
parser.add_argument('--port', type=str, default='5000')  
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')  
parser.add_argument('--gpu_mem_util', type=float, default=0.7,  
                    help='GPU memory utilization (lower for 4 GPU setup)')  
parser.add_argument('--tensor_parallel_size', type=int, default=4,  
                    help='Number of GPUs for tensor parallelism')  
args = parser.parse_args()  
  
# ------------------------ Timeout Utility --------------------------- #  
@stopit.threading_timeoutable(default='TIMED_OUT')  
def grade_answer_with_timeout(res1, res2):  
    return grade_answer(res1, res2)  
  
# ---------------------------- Flask Application --------------------------- #  
app = Flask(__name__)  
  
@app.route('/hello', methods=['GET'])  
def hello():  
    pause_event.set()  
    torch.cuda.synchronize()  
  
    name = request.args.get('name', 'None')  
    print(f'[server] Received request for task file: {name}')  
  
    with open(name, 'r') as f:  
        data = json.load(f)  
    os.remove(name)  
  
    questions = [item.get('question', '') for item in data]  
    answers   = [item.get('answer',   '') for item in data]  
  
    valid_indices, valid_questions, valid_answers, valid_chats = [], [], [], []  
    for i, (q, a) in enumerate(zip(questions, answers)):  
        if q and a:  
            valid_indices.append(i)  
            valid_questions.append(q)  
            valid_answers.append(a)  
            valid_chats.append([  
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},  
                {'role': 'user',   'content': q}  
            ])  
    print('[server] Valid chat prompts have been prepared.')  
  
    if valid_chats:  
        if tokenizer.chat_template:  
            prompts = [  
                tokenizer.apply_chat_template(chat, tokenize=False,  
                                              add_generation_prompt=True, add_special_tokens=True)  
                for chat in valid_chats  
            ]  
        else:  
            prompts = [  
                'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content']  
                for chat in valid_chats  
            ]  
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)  
    else:  
        responses = []  
    print('[server] Generation completed.')  
  
    def process_single(question, golden_answer, response):  
        results = [extract_boxed_content(out.text) for out in response.outputs]  
        answer_counts = {}  
          
        for res in results:  
            if not res: continue  
            matched = False  
              
            for exist_ans in list(answer_counts.keys()):  
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):  
                    answer_counts[exist_ans] += 1  
                    matched = True  
                    break  
                  
                try:  
                    is_match = False  
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)  
                    if match_result_1 == 'TIMED_OUT':  
                        print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")  
                    elif match_result_1:  
                        is_match = True  
  
                    if not is_match:  
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)  
                        if match_result_2 == 'TIMED_OUT':  
                            print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")  
                        elif match_result_2:  
                            is_match = True  
                      
                    if is_match:  
                        answer_counts[exist_ans] += 1  
                        matched = True  
                        break  
  
                except Exception as e:  
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")  
                    continue  
              
            if not matched:  
                answer_counts[res] = 1  
  
        if not answer_counts:  
            majority_ans, max_count = '', 0  
        else:  
            majority_ans = max(answer_counts, key=answer_counts.get)  
            max_count = answer_counts[majority_ans]  
  
        score = max_count / len(results) if results else 0.0  
  
        return {  
            'question': question,  
            'answer':   majority_ans,  
            'score':    score if majority_ans == golden_answer and score > 0.1 else 0,  
            'results':  results  
        }  
  
    results_all = []  
    response_idx = 0  
    for q, a in zip(questions, answers):  
        try:  
            if q and a:  
                response = responses[response_idx]  
                response_idx += 1  
                item = process_single(q, a, response)  
                results_all.append(item)  
            else:  
                results_all.append({'question': q, 'answer': a, 'score': -1, 'results': []})  
        except Exception as e:  
            print(f'[server] CRITICAL: An unhandled error occurred while processing question: {q}')  
            print(f'[server] Error details: {e}')  
            results_all.append({  
                'question': q,  
                'answer':   a,  
                'score':    -1,  
                'results':  [],  
                'error':    f'unhandled exception in process_single: {str(e)}'  
            })  
    print('[server] All results have been processed.')  
  
    out_path = name.replace('.json', '_results.json')  
    with open(out_path, 'w') as f:  
        json.dump(results_all, f, indent=4)  
  
    pause_event.clear()  
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')  
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})  
  
# ------------------------- Main Application Entrypoint --------------------------- #  
if __name__ == '__main__':  
    stop_event = threading.Event()  
    pause_event = threading.Event()  
  
    def gpu_idle_worker():  
        print('[idle_worker] GPU idle worker started.')  
        running = True  
        while not stop_event.is_set():  
            if pause_event.is_set():  
                if running:  
                    print('[idle_worker] Paused.')  
                    running = False  
                time.sleep(0.1)  
                continue  
            else:  
                if not running:  
                    print('[idle_worker] Resumed.')  
                    running = True  
            try:  
                a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')  
                b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')  
                torch.matmul(a, b)  
                torch.cuda.synchronize()  
            except RuntimeError as e:  
                print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')  
                time.sleep(1)  
        print('[idle_worker] GPU idle worker stopped.')  
  
    print(f'[init] Loading model with {args.tensor_parallel_size} GPUs via tensor parallelism...')  
      
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)  
    model = vllm.LLM(  
        model=args.model_path,  
        tokenizer=args.model_path,  
        gpu_memory_utilization=args.gpu_mem_util,  
        tensor_parallel_size=args.tensor_parallel_size,  
    )  
      
    sample_params = vllm.SamplingParams(  
        max_tokens=4096,  
        temperature=1.0,  
        top_p=1.0,  
        top_k=40,  
        stop_token_ids=[tokenizer.eos_token_id],  
        n=10,  
    )  
      
    idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)  
    idle_thread.start()  
      
    try:  
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)  
    finally:  
        stop_event.set()  
        idle_thread.join()  
        print('[main] Application shutdown complete.')