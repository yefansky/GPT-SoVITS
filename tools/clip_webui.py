import argparse
import os
import time
import uuid
import librosa
import gradio as gr
import numpy as np
import soundfile
import json
import copy


import debugpy
debugpy.listen(("localhost", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached")

g_json_key_text = "text"
g_json_key_path = "wav_path"
g_load_file = ""
g_batch = 10
g_index = 0
g_max_json_index = 0
g_text_list = []
g_audio_list = []
g_checkbox_list = []
g_data_json = []
g_edit_area_items = []
g_search_query = None
g_filtered_indices = None

def reload_data(index, batch):
    global g_index, g_batch
    g_index, g_batch = index, batch
    
    if g_filtered_indices is not None:
        filtered_data = [g_data_json[i] for i in g_filtered_indices[index : index + batch]]
        return [{g_json_key_text: d[g_json_key_text], g_json_key_path: d[g_json_key_path]} 
                for d in filtered_data]
    else:
        datas = g_data_json[index : index + batch]
        return [{g_json_key_text: d[g_json_key_text], g_json_key_path: d[g_json_key_path]} 
                for d in datas]

def b_change_index(index, batch):
    global g_index, g_batch
    g_index, g_batch = index, batch
    datas = reload_data(index, batch)
    output = []
    
    display_index = g_filtered_indices[index] if g_filtered_indices else index
    
    for i, item in enumerate(datas):
        output.append({
            "__type__": "update", 
            "label": f"Text {display_index + i}", 
            "value": item[g_json_key_text],
            "visible": True
        })
    
    for _ in range(g_batch - len(datas)):
        output.append({
            "__type__": "update", 
            "label": "Text", 
            "value": "",
            "visible": False
        })
    
    for item in datas:
        output.append(item[g_json_key_path])
    for _ in range(g_batch - len(datas)):
        output.append(None)
    
    for _ in range(g_batch):
        output.append(False)
    
    return output

def b_next_index(index, batch):
    b_save_file()
    if (index + batch) <= g_max_json_index:
        return index + batch, *b_change_index(index + batch, batch)
    else:
        return index, *b_change_index(index, batch)

def b_previous_index(index, batch):
    b_save_file()
    if (index - batch) >= 0:
        return index - batch, *b_change_index(index - batch, batch)
    else:
        return 0, *b_change_index(0, batch)

def b_submit_change(*text_list):
    global g_data_json
    change = False
    for i, new_text in enumerate(text_list):
        if g_index + i <= g_max_json_index:
            new_text = new_text.strip() + " "
            if g_data_json[g_index + i][g_json_key_text] != new_text:
                g_data_json[g_index + i][g_json_key_text] = new_text
                change = True
    if change:
        b_save_file()
    return g_index, *b_change_index(g_index, g_batch)

def save_edit_area_to_shared(*checkbox_list):
    edit_area_data = []
    if g_edit_area_items:
        edit_area_data = [{
            "audio_path": item["path"],
            "text": item["text"]
        } for item in g_edit_area_items]
    else:
        for i, checkbox in enumerate(checkbox_list):
            if checkbox:
                if g_filtered_indices:
                    if g_index + i < len(g_filtered_indices):
                        data_index = g_filtered_indices[g_index + i]
                        data = g_data_json[data_index]
                else:
                    if g_index + i < len(g_data_json):
                        data = g_data_json[g_index + i]
                
                if data:
                    edit_area_data.append({
                        "audio_path": data[g_json_key_path],
                        "text": data[g_json_key_text].strip()
                    })
    
    if edit_area_data:
        with open("./shared_ref.json", "w", encoding="utf-8") as f:
            json.dump(edit_area_data[0] if len(edit_area_data) == 1 else {
                "audio_path": "merged_audio.wav",
                "text": " ".join([d["text"] for d in edit_area_data])
            }, f)

def search_text(query):
    global g_search_query, g_filtered_indices, g_index
    
    if not query or query.strip() == "":
        g_search_query = None
        g_filtered_indices = None
        g_index = 0
        return {"value": 0, "maximum": g_max_json_index, "__type__": "update"}, *b_change_index(0, g_batch)
    
    query = query.strip()
    g_search_query = query.lower()
    g_filtered_indices = [
        idx for idx, item in enumerate(g_data_json)
        if g_search_query in item[g_json_key_text].lower()
    ]
    g_index = 0
    
    return {"value": 0, "maximum": g_max_json_index, "__type__": "update"}, *b_change_index(0, g_batch)

def update_select_item(action, checkbox_values):
    global g_edit_area_items
    
    if action == "add_from_checkboxes":
        new_items = []
        for i, checked in enumerate(checkbox_values):
            if checked and g_index + i < len(g_data_json):
                item = g_data_json[g_index + i]
                # 防止重复添加
                if not any(existing["index"] == g_index + i for existing in g_edit_area_items):
                    new_items.append({
                        "text": item[g_json_key_text],
                        "path": item[g_json_key_path],
                        "index": g_index + i
                    })
        
        # 合并新项（不超过最大限制）
        MAX_ITEMS = 5
        combined = g_edit_area_items + new_items
        g_edit_area_items = combined[:MAX_ITEMS]
        
        if len(combined) > MAX_ITEMS:
            gr.Warning(f"剪辑区已满，只保留了前{MAX_ITEMS}项")
    
    return generate_edit_area_outputs()

def handle_edit_area_actions(action, *checkbox_states):
    global g_edit_area_items
    
    # 从checkbox_states获取所有checkbox的当前值
    checkbox_values = checkbox_states[:5]  # 假设最多5个编辑项
    
    if action == "remove":
        # 找出第一个选中的checkbox索引
        selected_index = next((i for i, checked in enumerate(checkbox_values) if checked), None)
        if selected_index is not None and selected_index < len(g_edit_area_items):
            g_edit_area_items.pop(selected_index)
            gr.Info(f"已移除第 {selected_index+1} 项")
            
    elif action == "clear":
        if g_edit_area_items:
            g_edit_area_items.clear()
            gr.Info("已清空剪辑区")
    
    elif action == "move_up":
        selected_index = next((i for i, checked in enumerate(checkbox_values) if checked), None)
        if selected_index is not None and selected_index > 0:
            g_edit_area_items[selected_index], g_edit_area_items[selected_index-1] = \
                g_edit_area_items[selected_index-1], g_edit_area_items[selected_index]
    
    elif action == "move_down":
        selected_index = next((i for i, checked in enumerate(checkbox_values) if checked), None)
        if selected_index is not None and selected_index < len(g_edit_area_items)-1:
            g_edit_area_items[selected_index], g_edit_area_items[selected_index+1] = \
                g_edit_area_items[selected_index+1], g_edit_area_items[selected_index]
    
    return generate_edit_area_outputs()

# 新增生成剪辑区输出的函数
def generate_edit_area_outputs():
    outputs = []
    for i in range(5):
        if i < len(g_edit_area_items):
            # Checkbox更新 (移除choices参数)
            outputs.append({
                "value": False,  # 默认不选中
                "__type__": "update"
            })
            # Textbox更新
            outputs.append({
                "value": g_edit_area_items[i]["text"],
                "__type__": "update"
            })
            # Audio更新
            outputs.append({
                "value": g_edit_area_items[i]["path"],
                "__type__": "update"
            })
        else:
            # 隐藏多余的项目
            outputs.append({"value": False, "__type__": "update"})
            outputs.append({"value": "",  "__type__": "update"})
            outputs.append({"value": None, "__type__": "update"})
    return outputs

def merge_edit_area_audio(interval):
    global g_edit_area_items
    if not g_edit_area_items:
        return None
    
    audio_list = []
    sample_rate = None
    timestamp = str(int(time.time()))
    os.makedirs("temp", exist_ok=True)
    output_path = os.path.join("temp", f"merged_{timestamp}.wav")
    merged_text = " ".join([item["text"].strip() for item in g_edit_area_items])
    
    for item in g_edit_area_items:
        data, sr = librosa.load(item["path"], sr=sample_rate, mono=True)
        sample_rate = sr
        if audio_list:
            silence = np.zeros(int(sample_rate * float(interval)))
            audio_list.append(silence)
        audio_list.append(data)
    
    merged_audio = np.concatenate(audio_list)
    soundfile.write(output_path, merged_audio, sample_rate)
    
    return merged_text, output_path 

def b_delete_audio(*checkbox_list):
    global g_data_json, g_index, g_max_json_index
    b_save_file()
    change = False
    for i, checkbox in reversed(list(enumerate(checkbox_list))):
        if g_index + i < len(g_data_json):
            if checkbox == True:
                g_data_json.pop(g_index + i)
                change = True

    g_max_json_index = len(g_data_json) - 1
    if g_index > g_max_json_index:
        g_index = g_max_json_index
        g_index = g_index if g_index >= 0 else 0
    if change:
        b_save_file()
    return {
        "value": g_index,
        "__type__": "update",
        "maximum": (g_max_json_index if g_max_json_index >= 0 else 0),
    }, *b_change_index(g_index, g_batch)

def b_invert_selection(*checkbox_list):
    new_list = [not item if item is True else True for item in checkbox_list]
    return new_list

def get_next_path(filename):
    base_dir = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    for i in range(100):
        new_path = os.path.join(base_dir, f"{base_name}_{str(i).zfill(2)}.wav")
        if not os.path.exists(new_path):
            return new_path
    return os.path.join(base_dir, f"{str(uuid.uuid4())}.wav")

def b_audio_split(audio_breakpoint, *checkbox_list):
    global g_data_json, g_max_json_index
    checked_index = []
    for i, checkbox in enumerate(checkbox_list):
        if checkbox == True and g_index + i < len(g_data_json):
            checked_index.append(g_index + i)
    if len(checked_index) == 1:
        index = checked_index[0]
        audio_json = copy.deepcopy(g_data_json[index])
        path = audio_json[g_json_key_path]
        data, sample_rate = librosa.load(path, sr=None, mono=True)
        audio_maxframe = len(data)
        break_frame = int(audio_breakpoint * sample_rate)

        if break_frame >= 1 and break_frame < audio_maxframe:
            audio_first = data[0:break_frame]
            audio_second = data[break_frame:]
            nextpath = get_next_path(path)
            soundfile.write(nextpath, audio_second, sample_rate)
            soundfile.write(path, audio_first, sample_rate)
            g_data_json.insert(index + 1, audio_json)
            g_data_json[index + 1][g_json_key_path] = nextpath
            b_save_file()

    g_max_json_index = len(g_data_json) - 1
    return {"value": g_index, "maximum": g_max_json_index, "__type__": "update"}, *b_change_index(g_index, g_batch)

def b_merge_audio(interval_r, *checkbox_list):
    global g_data_json, g_max_json_index
    b_save_file()
    checked_index = []
    audios_path = []
    audios_text = []
    for i, checkbox in enumerate(checkbox_list):
        if checkbox == True and g_index + i < len(g_data_json):
            checked_index.append(g_index + i)

    if len(checked_index) > 1:
        for i in checked_index:
            audios_path.append(g_data_json[i][g_json_key_path])
            audios_text.append(g_data_json[i][g_json_key_text])
        for i in reversed(checked_index[1:]):
            g_data_json.pop(i)

        base_index = checked_index[0]
        base_path = audios_path[0]
        g_data_json[base_index][g_json_key_text] = "".join(audios_text)

        audio_list = []
        l_sample_rate = None
        for i, path in enumerate(audios_path):
            data, sample_rate = librosa.load(path, sr=l_sample_rate, mono=True)
            l_sample_rate = sample_rate
            if i > 0:
                silence = np.zeros(int(l_sample_rate * interval_r))
                audio_list.append(silence)

            audio_list.append(data)

        audio_concat = np.concatenate(audio_list)
        soundfile.write(base_path, audio_concat, l_sample_rate)
        b_save_file()

    g_max_json_index = len(g_data_json) - 1
    return {"value": g_index, "maximum": g_max_json_index, "__type__": "update"}, *b_change_index(g_index, g_batch)

def b_save_json():
    with open(g_load_file, "w", encoding="utf-8") as file:
        for data in g_data_json:
            file.write(f"{json.dumps(data, ensure_ascii=False)}\n")

def b_save_list():
    with open(g_load_file, "w", encoding="utf-8") as file:
        for data in g_data_json:
            wav_path = data["wav_path"]
            speaker_name = data["speaker_name"]
            language = data["language"]
            text = data["text"]
            file.write(f"{wav_path}|{speaker_name}|{language}|{text}".strip() + "\n")

def b_load_json():
    global g_data_json, g_max_json_index
    with open(g_load_file, "r", encoding="utf-8") as file:
        g_data_json = file.readlines()
        g_data_json = [json.loads(line) for line in g_data_json]
        g_max_json_index = len(g_data_json) - 1

def b_load_list():
    global g_data_json, g_max_json_index
    g_data_json = []
    with open(g_load_file, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                g_data_json.append({
                    "wav_path": parts[0],
                    "speaker_name": parts[1],
                    "language": parts[2],
                    "text": "|".join(parts[3:]).strip()
                })
    g_max_json_index = len(g_data_json) - 1

def b_save_file():
    if g_load_format == "json":
        b_save_json()
    elif g_load_format == "list":
        b_save_list()

def b_load_file():
    if g_load_format == "json":
        b_load_json()
    elif g_load_format == "list":
        b_load_list()

def set_global(load_json, load_list, json_key_text, json_key_path, batch):
    global g_json_key_text, g_json_key_path, g_load_file, g_load_format, g_batch
    g_batch = int(batch)
    if load_json != "None":
        g_load_format = "json"
        g_load_file = load_json
    elif load_list != "None":
        g_load_format = "list"
        g_load_file = load_list
    else:
        g_load_format = "list"
        g_load_file = "demo.list"
    g_json_key_text = json_key_text
    g_json_key_path = json_key_path
    b_load_file()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Clipper WebUI")
    parser.add_argument("--load_list", required=True, help="source file, like demo.list")
    parser.add_argument("--is_share", default="False", help="whether webui is_share=True")
    parser.add_argument("--webui_port_clip", default=9870, help="webui port")
    parser.add_argument("--g_batch", default=10, help="max number of items to display")
    args = parser.parse_args()
    g_load_file = args.load_list
    g_batch = int(args.g_batch)
    b_load_list()

    with gr.Blocks() as demo:
        with gr.Row():
            search_box = gr.Textbox(label="搜索文本", placeholder="输入搜索内容...")
            btn_search = gr.Button("搜索")
            btn_add_to_edit = gr.Button("添加到剪辑区")
            btn_send_to_infer = gr.Button("发送到推理页")
        
        with gr.Row():
            index_slider = gr.Slider(minimum=0, maximum=g_max_json_index, value=g_index, step=1, label="Index", scale=3)
            
            btn_previous_index = gr.Button("上一页")
            btn_next_index = gr.Button("下一页")

        with gr.Row():
            with gr.Column():
                for _ in range(0, g_batch):
                    with gr.Row():
                        text = gr.Textbox(label="Text", visible=True, scale=5)
                        audio_output = gr.Audio(label="Output Audio", visible=True, scale=5)
                        audio_check = gr.Checkbox(label="Yes", show_label=True, info="Choose Audio", scale=1)
                        g_text_list.append(text)
                        g_audio_list.append(audio_output)
                        g_checkbox_list.append(audio_check)

        with gr.Row():
            batchsize_slider = gr.Slider(
                minimum=1, maximum=g_batch, value=g_batch, step=1, label="Batch Size", scale=3, interactive=False
            )
            # interval_slider = gr.Slider(minimum=0, maximum=2, value=0, step=0.01, label="Interval", scale=3)
            btn_theme_dark = gr.Button("Light Theme", link="?__theme=light", scale=1)
            btn_theme_light = gr.Button("Dark Theme", link="?__theme=dark", scale=1)
                
        selected_index = gr.State(value=-1)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 剪辑区")
                with gr.Group():
                    edit_area_container = []
                    g_edit_area_checkboxes = []
                    edit_radios = []
                    for i in range(5):
                        with gr.Row(visible=True):
                            cb = gr.Checkbox(label=f"片段 {i+1}", interactive=True)
                            text = gr.Textbox(label=f"编辑文本 {i+1}", value="", scale=5)
                            audio = gr.Audio(label=f"编辑音频 {i+1}", interactive=False, scale=5)
                            edit_area_container.append((cb, text, audio))
                            g_edit_area_checkboxes.append(cb)
                    
                    with gr.Row():
                        btn_remove_edit = gr.Button("移除选中项")
                        btn_move_up = gr.Button("上移")
                        btn_move_down = gr.Button("下移")
                        btn_clear_edit = gr.Button("清空列表")

                        interval_slider = gr.Slider(minimum=0, maximum=2, value=0.5, step=0.1, label="合并间隔(秒)", scale=3)
                        btn_merge_edit = gr.Button("合并剪辑区音频")

                    merged_audio_text = gr.Textbox(label="合并后的文本")
                    merged_audio_output = gr.Audio(label="合并后的音频")

        def create_checkbox_handler(total_checkboxes):
            def handler(*args):
                # args[0:-1]是各个checkbox的当前值
                # args[-1]是之前选中的index
                current_states = list(args[:-1])
                previous_index = args[-1]
                
                # 找出哪个checkbox发生了变化
                changed_index = None
                for i in range(total_checkboxes):
                    if current_states[i] != (i == previous_index):
                        changed_index = i
                        break
                
                # 确定新的选中状态
                new_index = -1
                new_states = [False] * total_checkboxes
                
                if changed_index is not None:
                    if current_states[changed_index]:  # 如果是选中操作
                        new_index = changed_index
                        new_states[changed_index] = True
                    # 如果是取消选中操作，保持new_index=-1
                
                return [new_index] + new_states
            return handler

        # 绑定事件 - 所有checkbox共享同一个handler
        total_checkboxes = len(g_edit_area_checkboxes)
        for cb in g_edit_area_checkboxes:
            cb.change(
                fn=create_checkbox_handler(total_checkboxes),
                inputs=[*g_edit_area_checkboxes, selected_index],
                outputs=[selected_index, *g_edit_area_checkboxes]
            )

        btn_previous_index.click(
            b_previous_index,
            inputs=[index_slider, batchsize_slider],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list]
        )

        btn_next_index.click(
            b_next_index,
            inputs=[index_slider, batchsize_slider],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list]
        )

        btn_search.click(
            fn=search_text,
            inputs=[search_box],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list]
        )

        btn_add_to_edit.click(
            fn=lambda *checks: update_select_item("add_from_checkboxes", checkbox_values=checks),
            inputs=[*g_checkbox_list],
            outputs=[*[comp for row in edit_area_container for comp in row]]
        )

        btn_remove_edit.click(
            fn=lambda *checks: handle_edit_area_actions("remove", *checks),
            inputs=gr.Textbox(visible=False),  # Workaround to get radio value
            outputs=[*[comp for row in edit_area_container for comp in row]]
        )

        btn_move_up.click(
            fn=lambda *checks: handle_edit_area_actions("move_up", *checks),
            inputs=gr.Textbox(visible=False),
            outputs=[*[comp for row in edit_area_container for comp in row]]
        )

        btn_move_down.click(
            fn=lambda *checks: handle_edit_area_actions("move_down",*checks),
            inputs=gr.Textbox(visible=False),
            outputs=[*[comp for row in edit_area_container for comp in row]]
        )

        btn_clear_edit.click(
            fn=lambda: handle_edit_area_actions("clear"),
            outputs=[*[comp for row in edit_area_container for comp in row]]
        )

        btn_merge_edit.click(
            fn=merge_edit_area_audio,
            inputs=[interval_slider],
            outputs=[merged_audio_text, merged_audio_output]
        )

        btn_send_to_infer.click(save_edit_area_to_shared, inputs=g_checkbox_list)

        demo.load(
            b_change_index,
            inputs=[index_slider, batchsize_slider],
            outputs=[*g_text_list, *g_audio_list, *g_checkbox_list]
        )

    demo.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=eval(args.is_share),
        server_port=int(args.webui_port_clip)
    )