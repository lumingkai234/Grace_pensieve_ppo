import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from argparse import ArgumentParser

def read_df(path):
    if os.path.exists(path):
        return pd.read_csv(path).query("frame_id < 16")
    else:
        print("Skip the file because", path, "does not exist!")
        return None

def quick_plot_size(df, label):
    if df is None:
        print("Skip", label, "during size plot because dataframe is None")
        return

    df = df.sort_values("size").reset_index(drop=True)
    if "ssim_db" not in df.columns:
        df["ssim_db"] = -10 * np.log10(1 - df["ssim"])
    plt.plot(df["size"], df["ssim_db"], label=label, marker='o')

def quick_plot_loss(df, label):
    if df is None:
        print("Skip", label, "during loss plot because dataframe is None")
        return
    df = df.sort_values("loss").reset_index(drop=True)
    if "ssim_db" not in df.columns:
        df["ssim_db"] = -10 * np.log10(1 - df["ssim"])
    plt.plot(df["loss"], df["ssim_db"], label=label, marker='o')

def interpolate_quality(df, target_size):
    """
    assume input df has loss, size, ssim
    """
    if df is None:
        print("Skip", label, "during quality interpolation because dataframe is None")
        return
    df = df.sort_values(['loss', 'size'])

    def group_interpolate(group):
        #print(group['size'], group['ssim'])
        return pd.Series({'ssim': np.interp(target_size, group['size'], group['ssim'])})

    result = df.groupby(["loss"]).apply(group_interpolate).reset_index()
    #print(result)
    result["ssim_db"] = -10 * np.log10(1 - result["ssim"])
    return result

def quick_plot_fec(df, target_size, fec_ratio):
    if df is None:
        print("Skip", label, "during fec plot because dataframe is None")
        return
    #print(f'H.26x size is {size}')
    #print(df.groupby(["loss", "model"]).mean().reset_index())
    idf = interpolate_quality(df.groupby(["loss", "model"]).mean().reset_index(), size * (1-fec_ratio))
    #print(idf)
    quality = float(idf["ssim_db"])
    #print(quality)
    x = [0, fec_ratio - 0.01, fec_ratio + 0.01]
    y = [quality, quality, 8]
    plt.plot(x, y, label=f"{fec_ratio*100:.1f}% FEC")

if __name__ == "__main__":

    parser = ArgumentParser(description='Grace')
    parser.add_argument('--lamda', type=int, default=2048, help='trade-off parameter')
    args = parser.parse_args()

    video_filter = "video.str.contains('video-6')"
    df_grace = read_df("grace/all.csv").query(video_filter)
    df_grace_IND = read_df("grace_IND/all.csv").query(video_filter)
    print(df_grace["video"].unique())
    df_265 = read_df("h265/all.csv").query(video_filter)
    df_264 = read_df("h264/all.csv").query(video_filter)
    df_pretrain = read_df("pretrained/all.csv").query(video_filter)
    df_error = None #read_df("error_concealment/all.csv").query(video_filter)

    ''' QUALITY VS SIZE CURVE '''
    fig = plt.figure()
    if df_grace is not None:
        #print(df_grace.query("nframes == 0").groupby("model_id").mean().reset_index())
        quick_plot_size(df_grace.query("nframes == 0").groupby("model_id").mean().reset_index(), "grace")
        #print(df_grace.query("nframes == 0").groupby("model_id").mean().reset_index())
        #print(df_grace.query("nframes == 0").groupby("model_id").mean().reset_index())
    if df_pretrain is not None:
        quick_plot_size(df_pretrain.query("nframes == 0").groupby("model_id").mean().reset_index(), "pretrained")
    if df_265 is not None:
        quick_plot_size(df_265.groupby("model").mean().reset_index(), "H.265")
    if df_264 is not None:
        quick_plot_size(df_264.groupby("model").mean().reset_index(), "H.264")
    plt.xlim(0, 30000)
    plt.grid()
    plt.legend(['Grace', 'pretrain', 'H.265', 'H.264'])
    plt.xlabel("Size (bytes)")
    plt.ylabel("SSIM (dB)")

    fig.savefig("ssim_size.png")
    

    ''' QUALITY VS LOSS CURVE '''
    size = float(df_grace.query(f"nframes == 1 and model_id == {args.lamda}").mean()["size"])
    print(f'Vidieo mean size: {size}')
    for nframes in [1]:
        fig = plt.figure()
        if df_grace is not None:
            quick_plot_loss(interpolate_quality(df_grace.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size), "grace")
            #print(interpolate_quality(df_grace.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size))
            #ssim_db_values = interpolate_quality(df_grace.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size)["ssim_db"].tolist()
            #print(f'Grace : {ssim_db_values}')
        if df_grace_IND is not None:
            quick_plot_loss(interpolate_quality(df_grace_IND.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size), "grace_IND")
            #print(df_grace.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index())
            #print(interpolate_quality(df_grace_IND.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size))
            #ssim_db_values = interpolate_quality(df_grace_IND.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size)["ssim_db"].tolist()
            #print(f'Grace IND: {ssim_db_values}')
        if df_error is not None:
            quick_plot_loss(interpolate_quality(df_error.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size), "error concealment")
        #if df_pretrain is not None:
        #    quick_plot_loss(interpolate_quality(df_pretrain.query("nframes == @nframes").groupby(["loss", "model_id"]).mean().reset_index(), size), "pretrained")
        if df_265 is not None:
            quick_plot_fec(df_265, size, 0.2)
            quick_plot_fec(df_265, size, 0.5)
            quick_plot_fec(df_265, size, 0.7)
        if df_264 is not None:
            quick_plot_fec(df_264, size, 0.2)
            quick_plot_fec(df_264, size, 0.5)
            quick_plot_fec(df_264, size, 0.7)
        plt.grid()
        plt.legend(['Grace', 'Grace_IND', 'H.265 0.20 FEC', 'H.265 0.50 FEC', 'H.265 0.70 FEC', 'H.264 0.20 FEC', 'H.264 0.50 FEC', 'H.264 0.70 FEC'])
        plt.xlabel("Packet Loss Rate")
        plt.ylabel("SSIM (dB)")
        
        import sys
        # 获取当前文件的绝对路径
        current_file_path = os.path.abspath(__file__)

        # 获取父目录路径
        parent_dir = os.path.dirname(os.path.dirname(current_file_path))

        # 将父目录添加到 sys.path
        sys.path.append(parent_dir)
        from config_IND import config
        save_dir = f'./ssim_loss_lamda_{args.lamda}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(f"{save_dir}/ssim_loss_mv_{config.mv_important}_res_{config.res_important}.png")

