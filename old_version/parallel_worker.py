import numpy as np
import pandas as pd
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network
from brian2 import mV, ms, Hz, start_scope
from time import time

def run_freq_trial_batch(freq, neu_exc, data_package, params, n_trials=2):
    '''
    运行单个频率的所有trials（完全独立）
    
    Parameters
    ----------
    freq : int
        激活频率
    neu_exc : list
        要激活的神经元IDs
    data_package : dict
        包含所有必要数据（避免依赖全局变量）
    params : dict
        仿真参数
    n_trials : int
        trial数量
    
    Returns
    -------
    tuple : (freq, result_dict)
    '''
    
    # 解包数据
    root_ids = data_package['root_ids']
    df_conn = data_package['df_conn']
    pre_col = data_package['pre_col']
    post_col = data_package['post_col']
    weight_col = data_package['weight_col']
    nt_prob_cols = data_package['nt_prob_cols']
    
    # 重建flyid映射
    flyid2i = {int(rid): i for i, rid in enumerate(root_ids)}
    i2flyid = {i: int(rid) for i, rid in enumerate(root_ids)}
    n_neurons = len(root_ids)
    
    # === 简化的网络构建（避免调用复杂辅助函数）===
    start_scope()
    
    # 筛选连接（syn_count >= 5）
    df_filtered = df_conn[df_conn[weight_col] >= 5].copy()
    
    pre_ids = df_filtered[pre_col].values.astype(np.int64)
    post_ids = df_filtered[post_col].values.astype(np.int64)
    weights_raw = df_filtered[weight_col].values.astype(np.float64)
    
    # 映射到索引
    valid_ids_set = set(flyid2i.keys())
    valid_mask = np.array([
        (int(p) in valid_ids_set and int(q) in valid_ids_set) 
        for p, q in zip(pre_ids, post_ids)
    ])
    
    i_pre = np.array([flyid2i[int(p)] for p in pre_ids[valid_mask]])
    i_post = np.array([flyid2i[int(p)] for p in post_ids[valid_mask]])
    weights = weights_raw[valid_mask]
    
    # 简化：全部当兴奋性（忽略GABA判定）
    weights_final = weights * float(params['w_syn'] / mV) * mV
    
    # 创建神经元和突触
    neu = NeuronGroup(
        N=n_neurons,
        model=params['eqs'],
        method='linear',
        threshold=params['eq_th'],
        reset=params['eq_rst'],
        refractory='rfc',
        namespace=params,
    )
    neu.v = params['v_0']
    neu.g = 0
    neu.rfc = params['t_rfc']
    
    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=params['t_dly'])
    syn.connect(i=i_pre, j=i_post)
    syn.w = weights_final
    
    # === 运行trials ===
    all_spike_data = []
    
    for trial_idx in range(n_trials):
        # 每次重置神经元状态
        neu.v = params['v_0']
        neu.g = 0
        
        # Spike monitor
        spk_mon = SpikeMonitor(neu)
        
        # Poisson输入
        pois = []
        for nid in neu_exc:
            if nid in flyid2i:
                i = flyid2i[nid]
                p = PoissonInput(
                    target=neu[i],
                    target_var='v',
                    N=1,
                    rate=freq * Hz,
                    weight=params['w_syn'] * params['f_poi']
                )
                neu[i].rfc = 0 * ms
                pois.append(p)
        
        # 运行
        net = Network(neu, syn, spk_mon, *pois)
        net.run(params['t_run'])
        
        # 提取结果
        if len(spk_mon.i) > 0:
            trial_df = pd.DataFrame({
                't': np.array(spk_mon.t),
                'trial': trial_idx,
                'neuron_idx': np.array(spk_mon.i)
            })
            trial_df['flywire_id'] = trial_df['neuron_idx'].map(i2flyid)
            all_spike_data.append(trial_df[['t', 'trial', 'flywire_id']])
    
    # 合并结果
    if all_spike_data:
        df_result = pd.concat(all_spike_data, ignore_index=True)
    else:
        df_result = pd.DataFrame(columns=['t', 'trial', 'flywire_id'])
    
    n_spikes = len(df_result)
    n_active = df_result['flywire_id'].nunique() if n_spikes > 0 else 0
    
    return (freq, {
        'n_spikes': n_spikes,
        'n_active': n_active,
        'df': df_result
    })