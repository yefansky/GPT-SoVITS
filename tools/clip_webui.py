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
debugpy.listen(("localhost", 5678))  # 监听 localhost 的 5678 端口
print("Waiting for debugger to attach...")
debugpy.wait_for_client()  # 等待调试器 attach
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
g_selected_items = []
g_search_query = None
g_filtered_indices = None


def reload_data(index, batch):
    global g_index, g_batch
    g_index, g_batch = index, batch
    
    if g_filtered_indices != None:
        # 使用过滤后的索引
        filtered_data = [g_data_json[i] for i in g_filtered_indices[index : index + batch]]
        return [{g_json_key_text: d[g_json_key_text], g_json_key_path: d[g_json_key_path]} 
                for d in filtered_data]
    else:
        # 原始数据
        datas = g_data_json[index : index + batch]
        return [{g_json_key_text: d[g_json_key_text], g_json_key_path: d[g_json_key_path]} 
                for d in datas]


def b_change_index(index, batch):
    global g_index, g_batch
    g_index, g_batch = index, batch
    datas = reload_data(index, batch)
    output = []
    
    # 计算实际显示的索引
    display_index = g_filtered_indices[index] if g_filtered_indices else index
    
    for i, item in enumerate(datas):
        output.append({
            "__type__": "update", 
            "label": f"Text {display_index + i}", 
            "value": item[g_json_key_text],
            "visible": True
        })
    
    # 填充剩余位置
    for _ in range(g_batch - len(datas)):
        output.append({
            "__type__": "update", 
            "label": "Text", 
            "value": "",
            "visible": False
        })
    
    # 音频输出
    for item in datas:
        output.append(item[g_json_key_path])
    for _ in range(g_batch - len(datas)):
        output.append(None)
    
    # 复选框状态
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
    # 优先使用选中区的项目
    if g_selected_items:
        selected_data = [{
            "audio_path": item["path"],
            "text": item["text"]
        } for item in g_selected_items]
    else:
        # 回退到复选框选择
        for i, checkbox in enumerate(checkbox_list):
            if checkbox:
                # 获取正确的索引（考虑过滤情况）
                if g_filtered_indices:
                    # 确保索引在过滤范围内
                    if g_index + i < len(g_filtered_indices):
                        data_index = g_filtered_indices[g_index + i]
                        data = g_data_json[data_index]
                else:
                    # 原始情况
                    if g_index + i < len(g_data_json):
                        data = g_data_json[g_index + i]
                
                if data:  # 确保data已获取
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
        # 清除搜索状态
        g_search_query = None
        g_filtered_indices = None
        g_index = 0
        return {"value": 0, "maximum": g_max_json_index, "__type__": "update"}, *b_change_index(0, g_batch)
    
    query = query.strip()
    # 执行搜索
    g_search_query = query.lower()
    g_filtered_indices = [
        idx for idx, item in enumerate(g_data_json)
        if g_search_query in item[g_json_key_text].lower()
    ]
    g_index = 0
    
    # 返回更新后的界面状态，第一个值更新slider
    return {"value": 0, "maximum": g_max_json_index, "__type__": "update"}, *b_change_index(0, g_batch)

def update_selected_items(action, item_index=None, new_order=None, checkbox_values=None):
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
    elif action == "remove":
        if item_index is not None and 0 <= item_index < len(g_selected_items):
            g_selected_items.pop(item_index)
    elif action == "reorder" and new_order is not None:
        g_selected_items = [g_selected_items[i] for i in new_order]
    elif action == "clear":
        g_selected_items = []
    
    # 返回格式化后的选中项用于显示
    return [[item["text"], item["path"]] for item in g_selected_items]

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
        if audio_list:  # 添加间隔
            silence = np.zeros(int(sample_rate * float(interval)))
            audio_list.append(silence)
        audio_list.append(data)
    
    merged_audio = np.concatenate(audio_list)
    soundfile.write(output_path, merged_audio, sample_rate)
    
    # 返回音频路径和合并后的文本
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
    # return gr.Slider(value=g_index, maximum=(g_max_json_index if g_max_json_index>=0 else 0)), *b_change_index(g_index, g_batch)
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
    # return gr.Slider(value=g_index, maximum=g_max_json_index), *b_change_index(g_index, g_batch)
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

    # return gr.Slider(value=g_index, maximum=g_max_json_index), *b_change_index(g_index, g_batch)
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
            if len(parts) >= 4:  # 至少包含wav_path|speaker_name|language|text
                g_data_json.append({
                    "wav_path": parts[0],
                    "speaker_name": parts[1],
                    "language": parts[2],
                    "text": "|".join(parts[3:]).strip()  # 处理文本中可能包含的|字符
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
                
        # 修改后的选中区代码
        with gr.Row():
            with gr.Column():
                # 原Dataframe保持两列（不需要试听列）
                selected_items_display = gr.Dataframe(
                    headers=["文本", "路径"],
                    datatype=["str", "str"],
                    label="已选中项目 (可拖动排序)",
                    interactive=True,
                    row_count=5,
                    wrap=True
                )
                
                # 新增：为每行创建独立的音频组件
                audio_preview_rows = []
                for _ in range(5):  # 与row_count=5保持一致
                    with gr.Row(visible=False):  # 初始隐藏
                        gr.Textbox(visible=False)  # 隐藏文本占位
                        gr.Textbox(visible=False)  # 隐藏路径占位
                        audio = gr.Audio(label="试听", interactive=False)
                        audio_preview_rows.append(audio)
                
                # 操作按钮保持不变
                with gr.Row():
                    btn_remove_selected = gr.Button("移除选中项")
                    btn_move_up = gr.Button("上移")
                    btn_move_down = gr.Button("下移")
                    btn_clear_selected = gr.Button("清空列表")
                merged_audio_output = gr.Audio(label="合并后的音频", visible=False)

        btn_previous_index.click(
            b_previous_index,
            inputs=[
                index_slider,
                batchsize_slider,
            ],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list],
        )

        btn_next_index.click(
            b_next_index,
            inputs=[
                index_slider,
                batchsize_slider,
            ],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list],
        )

        # 新增交互逻辑
        btn_search.click(
                    fn=search_text,
                    inputs=[search_box],
                    outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list]
                )
        
        btn_add_to_selected.click(
            fn=lambda *checks: update_selected_items("add_from_checkboxes", checkbox_values=checks),
            inputs=[*g_checkbox_list],
            outputs=[selected_items_display]
        )

        btn_remove_selected.click(
            fn=lambda: update_selected_items("remove", item_index=selected_items_display.selected_index),
            outputs=[selected_items_display, merged_audio_output]
        )

        btn_move_up.click(
            fn=lambda: update_selected_items("reorder", new_order=[
                i-1 if i == selected_items_display.selected_index else 
                i+1 if i == selected_items_display.selected_index-1 else 
                i for i in range(len(g_selected_items))
            ]) if selected_items_display.selected_index and selected_items_display.selected_index > 0 else g_selected_items,
            outputs=[selected_items_display, merged_audio_output]
        )

        btn_move_down.click(
            fn=lambda: update_selected_items("reorder", new_order=[
                i+1 if i == selected_items_display.selected_index else 
                i-1 if i == selected_items_display.selected_index+1 else 
                i for i in range(len(g_selected_items))
            ]) if selected_items_display.selected_index is not None and 
                    selected_items_display.selected_index < len(g_selected_items)-1 else g_selected_items,
            outputs=[selected_items_display, merged_audio_output]
        )

        btn_clear_selected.click(
            fn=lambda: update_selected_items("clear"),
            outputs=[selected_items_display, merged_audio_output]
        )

        btn_merge_selected.click(
            fn=merge_selected_audio,
            inputs=[interval_slider],
            outputs=[merged_audio_output]
        )
        
        # 复选框选中时添加到选中区
        for i, checkbox in enumerate(g_checkbox_list):
            checkbox.change(
                fn=lambda x, idx=i: update_selected_items("add", g_index + idx) if x else None,
                inputs=[checkbox],
                outputs=[selected_items_display]
            )
        
        btn_remove_selected.click(
            fn=lambda: update_selected_items("remove", selected_items_display.selected_index),
            outputs=[selected_items_display]
        )
        
        btn_clear_selected.click(
            fn=lambda: update_selected_items("clear"),
            outputs=[selected_items_display]
        )
        
        btn_send_to_infer.click(save_selected_to_shared, inputs=g_checkbox_list)

        demo.load(
            b_change_index,
            inputs=[
                index_slider,
                batchsize_slider,
            ],
            outputs=[*g_text_list, *g_audio_list, *g_checkbox_list],
        )

    demo.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        # quiet=True,
        share=eval(args.is_share),
        server_port=int(args.webui_port_clip),
    )
