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

"""
import debugpy
debugpy.listen(("localhost", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached")
"""

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
g_selected_items = []
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

def save_selected_to_shared(*checkbox_list):
    selected_data = []
    if g_selected_items:
        selected_data = [{
            "audio_path": item["path"],
            "text": item["text"]
        } for item in g_selected_items]
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
                    selected_data.append({
                        "audio_path": data[g_json_key_path],
                        "text": data[g_json_key_text].strip()
                    })
    
    if selected_data:
        with open("./shared_ref.json", "w", encoding="utf-8") as f:
            json.dump(selected_data[0] if len(selected_data) == 1 else {
                "audio_path": "merged_audio.wav",
                "text": " ".join([d["text"] for d in selected_data])
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

def update_selected_items(action, item_index=None, new_order=None, checkbox_values=None, radio_value=None):
    global g_selected_items
    if action == "add_from_checkboxes":
        g_selected_items = []
        for i, checked in enumerate(checkbox_values):
            if checked and g_index + i < len(g_data_json):
                item = g_data_json[g_index + i]
                g_selected_items.append({
                    "text": item[g_json_key_text],
                    "path": item[g_json_key_path],
                    "index": g_index + i
                })
    elif action == "add":
        if checkbox_values and len(g_selected_items) < 5:
            idx = checkbox_values
            if idx < len(g_data_json):
                item = g_data_json[idx]
                if not any(item["index"] == idx for item in g_selected_items):
                    g_selected_items.append({
                        "text": item[g_json_key_text],
                        "path": item[g_json_key_path],
                        "index": idx
                    })
    elif action == "remove":
        if item_index is not None and 0 <= item_index < len(g_selected_items):
            g_selected_items.pop(item_index)
    elif action == "reorder" and radio_value is not None:
        selected_idx = int(radio_value.split()[-1]) - 1  # Extract index from "Item X"
        if selected_idx < 0 or selected_idx >= len(g_selected_items):
            return update_selected_items("no_action")
        new_order = list(range(len(g_selected_items)))
        if new_order and 0 <= selected_idx < len(new_order):
            if item_index == "up" and selected_idx > 0:
                new_order[selected_idx], new_order[selected_idx - 1] = new_order[selected_idx - 1], new_order[selected_idx]
            elif item_index == "down" and selected_idx < len(new_order) - 1:
                new_order[selected_idx], new_order[selected_idx + 1] = new_order[selected_idx + 1], new_order[selected_idx]
        g_selected_items = [g_selected_items[i] for i in new_order]
    elif action == "clear":
        g_selected_items = []
    elif action == "no_action":
        pass
    
    outputs = []
    selected_radio_value = f"Item {1}" if g_selected_items else None  # Default to first item if any
    for i in range(5):
        if i < len(g_selected_items):
            outputs.append({
                "choices": [f"Item {i+1}"],
                "value": selected_radio_value if f"Item {i+1}" == selected_radio_value else None,
                "visible": True,
                "__type__": "update"
            })
            outputs.append({
                "value": g_selected_items[i]["text"],
                "visible": True,
                "__type__": "update"
            })
            outputs.append({
                "value": g_selected_items[i]["path"],
                "visible": True,
                "__type__": "update"
            })
        else:
            outputs.append({
                "choices": [],
                "value": None,
                "visible": False,
                "__type__": "update"
            })
            outputs.append({
                "value": "",
                "visible": False,
                "__type__": "update"
            })
            outputs.append({
                "value": None,
                "visible": False,
                "__type__": "update"
            })
    return outputs

def merge_selected_audio(interval):
    global g_selected_items
    if not g_selected_items:
        return None
    
    audio_list = []
    sample_rate = None
    timestamp = str(int(time.time()))
    os.makedirs("temp", exist_ok=True)
    output_path = os.path.join("temp", f"merged_{timestamp}.wav")
    merged_text = " ".join([item["text"].strip() for item in g_selected_items])
    
    for item in g_selected_items:
        data, sr = librosa.load(item["path"], sr=sample_rate, mono=True)
        sample_rate = sr
        if audio_list:
            silence = np.zeros(int(sample_rate * float(interval)))
            audio_list.append(silence)
        audio_list.append(data)
    
    merged_audio = np.concatenate(audio_list)
    soundfile.write(output_path, merged_audio, sample_rate)
    
    return {
        "audio_path": output_path,
        "text": merged_text
    }

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
            btn_add_to_selected = gr.Button("添加到选中区")
            btn_merge_selected = gr.Button("合并选中音频")
            btn_send_to_infer = gr.Button("发送到推理页")
        
        with gr.Row():
            index_slider = gr.Slider(minimum=0, maximum=g_max_json_index, value=g_index, step=1, label="Index", scale=3)
            interval_slider = gr.Slider(minimum=0, maximum=2, value=0.5, step=0.1, label="合并间隔(秒)", scale=3)
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
            interval_slider = gr.Slider(minimum=0, maximum=2, value=0, step=0.01, label="Interval", scale=3)
            btn_theme_dark = gr.Button("Light Theme", link="?__theme=light", scale=1)
            btn_theme_light = gr.Button("Dark Theme", link="?__theme=dark", scale=1)
                
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Selected Items")
                with gr.Group():
                    selected_items_container = []
                    selected_radios = []
                    for i in range(5):
                        with gr.Row(visible=True):
                            radio = gr.Radio(
                                choices=[f"Item {i+1}"],
                                value=None,
                                show_label=False,
                                interactive=True,
                                scale=1
                            )
                            text = gr.Textbox(label=f"Selected Text {i+1}", value="", scale=5)
                            audio = gr.Audio(label=f"Selected Audio {i+1}", interactive=False, scale=5)
                            selected_items_container.append((radio, text, audio))
                            selected_radios.append(radio)
                    
                    with gr.Row():
                        btn_remove_selected = gr.Button("移除选中项")
                        btn_move_up = gr.Button("上移")
                        btn_move_down = gr.Button("下移")
                        btn_clear_selected = gr.Button("清空列表")
                    merged_audio_output = gr.Audio(label="合并后的音频", visible=False)

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

        btn_add_to_selected.click(
            fn=lambda *checks: update_selected_items("add_from_checkboxes", checkbox_values=checks),
            inputs=[*g_checkbox_list],
            outputs=[*[comp for pair in selected_items_container for comp in pair]]
        )

        btn_remove_selected.click(
            fn=lambda *radios: update_selected_items(
                "remove", 
                item_index=[i for i, r in enumerate(radios) if r][0] if any(radios) else 0
            ),
            inputs=selected_radios,
            outputs=[*[comp for pair in selected_items_container for comp in pair]]
        )

        btn_move_up.click(
            fn=lambda *radios: update_selected_items(
                "reorder", 
                item_index="up", 
                radio_value=[r for r in radios if r][0] if any(radios) else None
            ),
            inputs=selected_radios,
            outputs=[*[comp for pair in selected_items_container for comp in pair]]
        )

        btn_move_down.click(
            fn=lambda *radios: update_selected_items(
                "reorder", 
                item_index="down", 
                radio_value=[r for r in radios if r][0] if any(radios) else None
            ),
            inputs=selected_radios,
            outputs=[*[comp for pair in selected_items_container for comp in pair]]
        )

        btn_clear_selected.click(
            fn=lambda: update_selected_items("clear"),
            outputs=[*[comp for pair in selected_items_container for comp in pair]]
        )

        btn_merge_selected.click(
            fn=merge_selected_audio,
            inputs=[interval_slider],
            outputs=[merged_audio_output]
        )

        for i, checkbox in enumerate(g_checkbox_list):
            checkbox.change(
                fn=lambda x, idx=i: update_selected_items("add", checkbox_values=g_index + idx) if x else update_selected_items("no_action"),
                inputs=[checkbox],
                outputs=[*[comp for pair in selected_items_container for comp in pair]]
            )

        btn_send_to_infer.click(save_selected_to_shared, inputs=g_checkbox_list)

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