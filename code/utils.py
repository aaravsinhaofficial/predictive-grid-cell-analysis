import numpy as np
import torch


def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in 
    params.npy file. 
    '''
    params = [
        'steps', str(options.sequence_length),
        'batch', str(options.batch_size),
        options.RNN_type,
        str(options.Ng),
        options.activation,
        'rf', str(options.place_cell_rf),
        'DoG', str(options.DoG),
        'periodic', str(options.periodic),
        'lr', str(options.learning_rate),
        'weight_decay', str(options.weight_decay),
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID


def get_2d_sort(x1,x2):
    """
    Reshapes x1 and x2 into square arrays, and then sorts
    them such that x1 increases downward and x2 increases
    rightward. Returns the order.
    """
    n = int(np.round(np.sqrt(len(x1))))
    total_order = x1.argsort()
    total_order = total_order.reshape(n,n)
    for i in range(n):
        row_order = x2[total_order.ravel()].reshape(n,n)[i].argsort()
        total_order[i] = total_order[i,row_order]
    total_order = total_order.ravel()
    return total_order


def dft(N,real=False,scale='sqrtn'):
    if not real:
        return scipy.linalg.dft(N,scale)
    else:
        cosines = np.cos(2*np.pi*np.arange(N//2+1)[None,:]/N*np.arange(N)[:,None])
        sines = np.sin(2*np.pi*np.arange(1,(N-1)//2+1)[None,:]/N*np.arange(N)[:,None])
        if N%2==0:
            cosines[:,-1] /= np.sqrt(2)
        F = np.concatenate((cosines,sines[:,::-1]),1)
        F[:,0] /= np.sqrt(N)
        F[:,1:] /= np.sqrt(N/2)
        return F


def skaggs_power(Jsort):
    F = dft(int(np.sqrt(N)), real=True)
    F2d = F[:,None,:,None]*F[None,:,None,:]

    F2d_unroll = np.reshape(F2d, (N, N))

    F2d_inv = F2d_unroll.conj().T
    Jtilde = F2d_inv.dot(Jsort).dot(F2d_unroll)

    return (Jtilde[1,1]**2 + Jtilde[-1,-1]**2) / (Jtilde**2).sum()


def skaggs_power_2(Jsort):
    J_square = np.reshape(Jsort, (n,n,n,n))
    Jmean = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            Jmean += np.roll(np.roll(J_square[i,j], -i, axis=0), -j, axis=1)

#     Jmean[0,0] = np.max(Jmean[1:,1:])
    Jmean = np.roll(np.roll(Jmean, n//2, axis=0), n//2, axis=1)
    Jtilde = np.real(np.fft.fft2(Jmean))
    
    Jtilde[0,0] = 0
    sk_power = Jtilde[1,1]**2 + Jtilde[0,1]**2 + Jtilde[1,0]**2
    sk_power += Jtilde[-1,-1]**2 + Jtilde[0,-1]**2 + Jtilde[-1,0]**2
    sk_power /= (Jtilde**2).sum()
    
    return sk_power


def calc_err():
    inputs, _, pos = next(gen)
    pred = model(inputs)
    pred_pos = place_cells.get_nearest_cell_pos(pred)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum((pos - pred_pos)**2, axis=-1)))

from visualize import compute_ratemaps, plot_ratemaps


def compute_variance(res, n_avg):
    
    activations, rate_map, g, pos = compute_ratemaps(model, data_manager, options, res=res, n_avg=n_avg)

    counts = np.zeros([res,res])
    variance = np.zeros([res,res])

    x_all = (pos[:,0] + options['box_width']/2) / options['box_width'] * res
    y_all = (pos[:,1] + options['box_height']/2) / options['box_height'] * res
    for i in tqdm(range(len(g))):
        x = int(x_all[i])
        y = int(y_all[i])
        if x >=0 and x < res and y >=0 and y < res:
            counts[x, y] += 1
            variance[x, y] += np.linalg.norm(g[i] - activations[:, x, y]) / np.linalg.norm(g[i]) / np.linalg.norm(activations[:,x,y])

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                variance[x, y] /= counts[x, y]
                
    return variance


def load_trained_weights(model, trainer, weight_dir):
    ''' Load weights stored as a .npy file (for github)'''

    # Train for a single step to initialize weights
    trainer.train(n_epochs=1, n_steps=1, save=False)

    # Load weights from npy array
    weights = np.load(weight_dir, allow_pickle=True)
    model.set_weights(weights)
    print('Loaded trained weights.')


def load_example_npy_weights_into_model(model: torch.nn.Module, npy_path: str):
    '''
    Load weights from models/example_trained_weights.npy (TF-era format)
    into the current PyTorch RNN model.

    Expected layout in the .npy file (object array of length 4):
      [0] decoder (Np, Ng)
      [1] input (2, Ng)  -> transposed to (Ng, 2) for torch RNN.weight_ih_l0
      [2] recurrent (Ng, Ng)
      [3] encoder (Ng, Np)

    The model must be constructed with matching Np/Ng.
    '''
    weights = np.load(npy_path, allow_pickle=True)
    if not isinstance(weights, np.ndarray) or weights.dtype != object or len(weights) != 4:
        raise ValueError('Unexpected weights format. Expected object ndarray of length 4.')

    w_dec = weights[0]  # (Np, Ng)
    w_in = weights[1]   # (2, Ng) -> transpose
    w_rec = weights[2]  # (Ng, Ng)
    w_enc = weights[3]  # (Ng, Np)

    Np_w, Ng_w = w_dec.shape
    Ng_w_in = w_in.shape[1]
    Ng_w_rec0, Ng_w_rec1 = w_rec.shape
    Ng_w_enc, Np_w_enc = w_enc.shape

    # Basic validations
    if Ng_w_in != Ng_w or Ng_w_rec0 != Ng_w or Ng_w_rec1 != Ng_w or Ng_w_enc != Ng_w or Np_w_enc != Np_w:
        raise ValueError('Inconsistent shapes in weight file.')

    if getattr(model, 'Ng', None) != Ng_w or getattr(model, 'Np', None) != Np_w:
        raise ValueError(f'Model dims (Np={getattr(model, "Np", None)}, Ng={getattr(model, "Ng", None)}) do not match weights (Np={Np_w}, Ng={Ng_w}). Set --Np {Np_w} --Ng {Ng_w} when constructing the model.')

    device = next(model.parameters()).device
    with torch.no_grad():
        # decoder: (Np, Ng)
        model.decoder.weight.copy_(torch.from_numpy(w_dec).to(device).float())

        # RNN input weights: torch expects (Ng, 2)
        model.RNN.weight_ih_l0.copy_(torch.from_numpy(w_in.T).to(device).float())

        # RNN recurrent weights: (Ng, Ng)
        model.RNN.weight_hh_l0.copy_(torch.from_numpy(w_rec).to(device).float())

        # encoder: (Ng, Np)
        model.encoder.weight.copy_(torch.from_numpy(w_enc).to(device).float())

    print(f'Loaded TF-era .npy weights into PyTorch model (Np={Np_w}, Ng={Ng_w}).')
