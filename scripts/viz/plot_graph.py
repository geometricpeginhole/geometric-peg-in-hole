import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


if __name__ == '__main__':
    data_wrist = pd.read_csv('results/wrist_view.csv')
    data_wrist = pd.melt(data_wrist, 'Model', var_name='Task', value_name='Success Rate')
    data_wrist.loc[:, 'Task'] = data_wrist.loc[:, 'Task'].str.replace(' ', '\n')
    data_wrist.loc[:, 'Wrist'] = 'Wrist'
    # plot all
    # sns.barplot(data=data_wrist, x='Task', hue='Model', y='Success Rate')
    # plt.show()

    # plot resnet18 all variations
    data1 = data_wrist[data_wrist['Model'].str.contains('Non-pretrained ResNet-18') & data_wrist['Task'].str.contains('(100)', regex=False)]
    plt.figure(figsize=(15, 2))
    ax = sns.barplot(data=data1, x='Task', y='Success Rate', hue='Model')
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel('')
    plt.title('ResNet-18 on All Task Variations')
    plt.savefig('results/resnet18-all.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot all models easy variations
    # data1 = data_wrist[data_wrist['Task'].str.contains('(100)', regex=False)]
    # plt.figure(figsize=(15, 2))
    # ax = sns.barplot(data=data1, x='Task', y='Success Rate', hue='Model')
    # for i in ax.containers:
    #     ax.bar_label(i, fontsize=7)
    # ax.set_ylim(0, 1.1)
    # fontP = FontProperties()
    # fontP.set_size('x-small')
    # legend = ax.legend(prop = fontP)
    # plt.setp(legend.get_title(),fontsize='x-small')

    # plt.title('All Models on Easy Task Variations')
    # plt.savefig('results/all-easy.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # plot data_wrist quantity variations
    data2 = data_wrist.loc[(data_wrist['Task'].str.len() > 8)].copy()
    data2.loc[:, 'Quantity'] = data2['Task'].str.extract(r'\((\d+)\)').values[:, 0]
    data2.loc[:, 'Task'] = data2.loc[:, 'Task'].str.split('\n').str[0]
    fig, axs = plt.subplots(ncols=3, figsize=(15, 2))
    ax = sns.barplot(data=data2.loc[data2['Model'].str.contains('Non-pretrained ResNet-18')], x='Task', y='Success Rate', hue='Quantity', ax=axs[0])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_title('ResNet-18')
    ax = sns.barplot(data=data2.loc[data2['Model'].str.contains('Non-pretrained ResNet-50')], x='Task', y='Success Rate', hue='Quantity', ax=axs[1])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_title('ResNet-50')
    ax = sns.barplot(data=data2.loc[data2['Model'].str.contains('CLIP ResNet-50')], x='Task', y='Success Rate', hue='Quantity', ax=axs[2])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_title('CLIP ResNet-50')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axs[0].legend([],[], frameon=False)
    axs[1].legend([],[], frameon=False)
    axs[2].legend([],[], frameon=False)
    axs[0].set_ylim(0, 1.1)
    axs[1].set_ylim(0, 1.1)
    axs[2].set_ylim(0, 1.1)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('')
    plt.savefig('results/data_quantity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot 1 variation
    plt.figure(figsize=(15, 2))
    data3 = data_wrist.loc[data_wrist.loc[:, 'Task'].str.contains('(100)', regex=False) & (data_wrist.loc[:, 'Task'].str.len() <= 8)]
    plt.figure(figsize=(15, 2))
    ax = sns.barplot(data=data3, x='Task', y='Success Rate', hue='Model')
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel('')
    # plt.title('All Models on 1 Variation')
    plt.legend(fontsize=8, bbox_to_anchor=(1, 1.3))
    plt.savefig('results/all-easy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot hard 100 variations
    data3 = data_wrist.loc[data_wrist.loc[:, 'Task'].str.contains('(100)', regex=False) & (data_wrist.loc[:, 'Task'].str.len() > 8)].copy().reset_index()
    data3.loc[:, 'Task'] = data3.loc[:, 'Task'].str.replace('\n', ' ')
    for model in data3['Model'].unique():
        print(model)
        print(data3.loc[data3['Model'] == model, 'Success Rate'].mean().round(2))
        data3.loc[len(data3)] = {'Model': model, 'Task': 'Average', 'Success Rate': data3.loc[data3['Model'] == model, 'Success Rate'].mean().round(2)}
    fig, axs = plt.subplots(nrows=3, figsize=(15, 5))
    ax = sns.barplot(data=data3, x='Task', y='Success Rate', hue='Model', ax=axs[0])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title('Num Demos During Training = 100')

    # plot hard 1000 variations
    data3 = data_wrist.loc[data_wrist.loc[:, 'Task'].str.contains('(1000)', regex=False) & (data_wrist.loc[:, 'Task'].str.len() > 8)].copy().reset_index()
    data3.loc[:, 'Task'] = data3.loc[:, 'Task'].str.replace('\n', ' ')
    for model in data3['Model'].unique():
        data3.loc[len(data3)] = {'Model': model, 'Task': 'Average', 'Success Rate': data3.loc[data3['Model'] == model, 'Success Rate'].mean().round(2)}
    ax = sns.barplot(data=data3, x='Task', y='Success Rate', hue='Model', ax=axs[1])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title('Num Demos During Training = 1000')

    # plot hard 10000 variations
    data3 = data_wrist.loc[data_wrist.loc[:, 'Task'].str.contains('(10000)', regex=False) & (data_wrist.loc[:, 'Task'].str.len() > 8)].copy().reset_index()
    data3.loc[:, 'Task'] = data3.loc[:, 'Task'].str.replace('\n', ' ')
    for model in data3['Model'].unique():
        data3.loc[len(data3)] = {'Model': model, 'Task': 'Average', 'Success Rate': data3.loc[data3['Model'] == model, 'Success Rate'].mean().round(2)}
    ax = sns.barplot(data=data3, x='Task', y='Success Rate', hue='Model', ax=axs[2])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title('Num Demos During Training = 10000')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, bbox_to_anchor=(1.13,0.65))
    axs[0].legend([],[], frameon=False)
    axs[1].legend([],[], frameon=False)
    axs[2].legend([],[], frameon=False)
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('')
    fig.tight_layout()
    plt.savefig('results/all-hard.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot pretrained model comparison
    data4 = data_wrist.loc[data_wrist.loc[:, 'Task'].str.contains('(100)', regex=False)].copy()
    fig, axs = plt.subplots(ncols=2, figsize=(15, 2))
    ax = sns.barplot(data=data4.loc[data4['Model'].str.contains('ResNet') & ~data4['Model'].str.contains('Non')], x='Task', y='Success Rate', hue='Model', ax=axs[0])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1, -0.1))
    ax = sns.barplot(data=data4.loc[data4['Model'].str.contains('ViT') & ~data4['Model'].str.contains('Non')], x='Task', y='Success Rate', hue='Model', ax=axs[1])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(0.1, -0.1))
    axs[0].set_title('Pretrained ResNet Models')
    axs[1].set_title('Pretrained ViT Models')
    
    plt.savefig('results/pre-all.png', dpi=300, bbox_inches='tight')
    plt.close()


    data_nowrist = pd.read_csv('results/top_view.csv')
    data_nowrist = pd.melt(data_nowrist, 'Model', var_name='Task', value_name='Success Rate')
    data_nowrist.loc[:, 'Task'] = data_nowrist.loc[:, 'Task'].str.replace(' ', '\n')
    data_nowrist.loc[:, 'Wrist'] = 'No Wrist'
    data_all = pd.concat([data_wrist, data_nowrist])

    # plot wrist vs nowrist of resnet-18 for all variations
    data5 = data_all.loc[data_all['Model'].str.contains('ResNet-18') & (data_all['Task'].str.len() > 8)].copy()
    fig, axs = plt.subplots(ncols=3, figsize=(15, 2))
    ax = sns.barplot(data=data5.loc[data5['Task'].str.contains('(100)', regex=False)], x='Task', y='Success Rate', hue='Wrist', ax=axs[0])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=8)
    ax = sns.barplot(data=data5.loc[data5['Task'].str.contains('(1000)', regex=False)], x='Task', y='Success Rate', hue='Wrist', ax=axs[1])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=8)
    ax = sns.barplot(data=data5.loc[data5['Task'].str.contains('(10000)', regex=False)], x='Task', y='Success Rate', hue='Wrist', ax=axs[2])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axs[0].legend([],[], frameon=False)
    axs[1].legend([],[], frameon=False)
    axs[2].legend([],[], frameon=False)
    axs[0].set_title('100 Episodes')
    axs[1].set_title('1000 Episodes')
    axs[2].set_title('10000 Episodes')
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('')
    plt.savefig('results/resnet18-wrist_nowrist.png', dpi=300, bbox_inches='tight')

    # plot nowrist vs. unfreeze of resnets for all variations
    data_unfreeze = pd.read_csv('results/top_view_unfreeze.csv')
    data_unfreeze = pd.melt(data_unfreeze, 'Model', var_name='Task', value_name='Success Rate')
    data_unfreeze.loc[:, 'Task'] = data_unfreeze.loc[:, 'Task'].str.replace(' ', '\n')
    data_unfreeze.loc[:, 'Wrist'] = 'No Wrist'
    data_unfreeze.loc[:, 'Freeze'] = 'Unfreeze'
    data_nowrist.loc[:, 'Freeze'] = 'Freeze'
    data_unfreeze = pd.concat([data_unfreeze, data_nowrist])
    data_unfreeze.loc[data_unfreeze['Model'].str.contains('Non'), 'Freeze'] = 'Unfreeze'
    fig, axs = plt.subplots(ncols=4, figsize=(15, 2))
    ax = sns.barplot(data=data_unfreeze.loc[data_unfreeze['Model'].str.contains('Non-pretrained ResNet-18')
                                            & (data_unfreeze['Task'].str.len() > 8)
                                            & ~data_unfreeze['Task'].str.contains('10000')],
                        x='Task', y='Success Rate', hue='Freeze', ax=axs[0])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=7)
    ax = sns.barplot(data=data_unfreeze.loc[data_unfreeze['Model'].str.contains('ImageNet ResNet-50')
                                            & (data_unfreeze['Task'].str.len() > 8)
                                            & ~data_unfreeze['Task'].str.contains('10000')],
                     x='Task', y='Success Rate', hue='Freeze', ax=axs[1])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=7)
    ax = sns.barplot(data=data_unfreeze.loc[data_unfreeze['Model'].str.contains('CLIP ResNet-50')
                                            & (data_unfreeze['Task'].str.len() > 8)
                                            & ~data_unfreeze['Task'].str.contains('10000')],
                        x='Task', y='Success Rate', hue='Freeze', ax=axs[2])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=7)
    ax = sns.barplot(data=data_unfreeze.loc[data_unfreeze['Model'].str.contains('R3M ResNet-50')
                                            & (data_unfreeze['Task'].str.len() > 8)
                                            & ~data_unfreeze['Task'].str.contains('10000')],
                        x='Task', y='Success Rate', hue='Freeze', ax=axs[3])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=7)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.55, -0.15))
    axs[0].legend([],[], frameon=False)
    axs[1].legend([],[], frameon=False)
    axs[2].legend([],[], frameon=False)
    axs[3].legend([],[], frameon=False)
    axs[0].set_title('Non-pretrained ResNet-18')
    axs[1].set_title('ImageNet ResNet-50')
    axs[2].set_title('CLIP ResNet-50')
    axs[3].set_title('R3M ResNet-50')
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[3].set_ylabel('')
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('')
    axs[3].set_xlabel('')
    plt.savefig('results/resnets-unfreeze.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot nowrist vs. unfreeze of vits for all variations
    fig, axs = plt.subplots(ncols=3, figsize=(15, 2))
    ax = sns.barplot(data=data_unfreeze.loc[data_unfreeze['Model'].str.contains('ImageNet ViT-B/16')
                                            & (data_unfreeze['Task'].str.len() > 8)
                                            & ~data_unfreeze['Task'].str.contains('10000')],
                     x='Task', y='Success Rate', hue='Freeze', ax=axs[0])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=8)
    ax = sns.barplot(data=data_unfreeze.loc[data_unfreeze['Model'].str.contains('CLIP ViT-B/16')
                                            & (data_unfreeze['Task'].str.len() > 8)
                                            & ~data_unfreeze['Task'].str.contains('10000')],
                        x='Task', y='Success Rate', hue='Freeze', ax=axs[1])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=8)
    ax = sns.barplot(data=data_unfreeze.loc[data_unfreeze['Model'].str.contains('MAE ViT-B/16')
                                            & (data_unfreeze['Task'].str.len() > 8)
                                            & ~data_unfreeze['Task'].str.contains('10000')],
                        x='Task', y='Success Rate', hue='Freeze', ax=axs[2])
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right')
    axs[0].legend([],[], frameon=False)
    axs[1].legend([],[], frameon=False)
    axs[2].legend([],[], frameon=False)
    axs[0].set_title('ImageNet ViT-B/16')
    axs[1].set_title('CLIP ViT-B/16')
    axs[2].set_title('MAE ViT-B/16')
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('')
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    plt.savefig('results/vits-unfreeze.png', dpi=300, bbox_inches='tight')
    plt.close()

    data_objects = pd.read_csv('results/objects.csv')
    data_objects = pd.melt(data_objects, 'Object Set', var_name='Task', value_name='Success Rate')
    # data_objects.loc[:, 'Task'] = data_objects.loc[:, 'Task'].str.replace(' ', '\n')
    # for model in data_objects['Task'].unique():
    #     data_objects.loc[len(data_objects)] = ['Average', model, data_objects.loc[data_objects['Task'] == model, 'Success Rate'].mean()]
    plt.figure(figsize=(12, 1))
    ax = sns.barplot(data=data_objects, x='Task', y='Success Rate', hue='Object Set')
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.legend(bbox_to_anchor=(1, 1.05))
    ax.tick_params(axis='x', labelsize=8)
    plt.savefig('results/objects.png', dpi=300, bbox_inches='tight')
    plt.close()

    data_objects = pd.read_csv('results/rotations.csv')
    data_objects = pd.melt(data_objects, 'Rotation-Loss', var_name='Task', value_name='Success Rate')
    # data_objects.loc[:, 'Task'] = data_objects.loc[:, 'Task'].str.replace(' ', '\n')
    # for model in data_objects['Task'].unique():
    #     data_objects.loc[len(data_objects)] = ['Average', model, data_objects.loc[data_objects['Task'] == model, 'Success Rate'].mean()]
    plt.figure(figsize=(15, 2))
    ax = sns.barplot(data=data_objects, x='Task', y='Success Rate', hue='Rotation-Loss')
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('')
    ax.legend(bbox_to_anchor=(1, 1.05))
    ax.tick_params(axis='x', labelsize=8)
    plt.savefig('results/rotations.png', dpi=300, bbox_inches='tight')
    plt.close()

    data_objects = pd.read_csv('results/real_world.csv')
    data_objects = pd.melt(data_objects, 'Model', var_name='Task', value_name='Success Rate')
    plt.figure(figsize=(15, 2))
    ax = sns.barplot(data=data_objects, x='Task', y='Success Rate')
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('')
    # ax.legend(bbox_to_anchor=(1, 1.05))
    ax.tick_params(axis='x', labelsize=8)
    plt.savefig('results/real_world.png', dpi=300, bbox_inches='tight')
    plt.close()

    data_objects = pd.read_csv('results/objects_yrzr.csv')
    data_objects = pd.melt(data_objects, 'Object Set', var_name='Num of Training Steps', value_name='Success Rate')
    plt.figure(figsize=(3, 1))
    ax = sns.lineplot(data=data_objects, x='Num of Training Steps', y='Success Rate', hue='Object Set')
    for i in ax.containers:
        ax.bar_label(i, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1, 1.05))
    ax.tick_params(axis='x', labelsize=8)
    plt.savefig('results/objects_yrzr.png', dpi=300, bbox_inches='tight')
    plt.close()