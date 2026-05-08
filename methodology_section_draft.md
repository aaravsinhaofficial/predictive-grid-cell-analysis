# Methodology

## Recurrent Neural Network Model

We trained recurrent neural networks (RNNs) to perform path integration in a simulated two-dimensional arena (2.2 x 2.2 m) with reflective boundaries. The network architecture follows Banino et al. (2018) and Cueva and Wei (2018), consisting of three components: (i) a linear encoder mapping place cell activations to a grid cell layer, (ii) a vanilla RNN with rectified linear unit (ReLU) nonlinearity operating on velocity inputs, and (iii) a linear decoder mapping grid cell activations back to place cell space.

The place cell population comprised N_p = 256 units with receptive fields modeled as difference-of-Gaussians (DoG; sigma = 0.12 m, surround scale ratio = 2), with centers uniformly distributed across the arena. Place cell activations were computed as softmax-normalized Gaussian responses to the agent's position, providing the supervised training signal.

The RNN hidden layer contained N_g = 4,096 units. At each time step, the network received a two-dimensional velocity input and maintained a hidden state h_t updated via the recurrence:

h_t = ReLU(W_hh * h_{t-1} + W_ih * v_t)

where W_hh denotes the recurrent weight matrix and W_ih the input weight matrix (both without bias). The initial hidden state h_0 was set by encoding the initial place cell activation through the encoder. The decoder produced predicted place cell activations, normalized via softmax.

The loss function combined cross-entropy between predicted and target place cell codes with L2 regularization on the recurrent weights:

L = -mean(y * log(softmax(predictions))) + lambda * ||W_hh||^2

where lambda = 1 x 10^{-6} was the weight decay coefficient. Networks were optimized using Adam with a learning rate of 1 x 10^{-4} over 5 epochs of 250 batches each (batch size = 100 trajectories of 20 time steps). Decoding error was computed as the Euclidean distance between the true position and the position estimated by averaging the centers of the k = 3 most active predicted place cells.

To ensure generality of our findings, we trained five independent networks with different random seeds (Seeds 0--4), all using identical hyperparameters.

## Trajectory Generation

Synthetic trajectories were generated as random walks with biologically inspired motion statistics. At each time step (dt = 0.02 s), the agent's forward speed was sampled from a Rayleigh distribution (b = 0.13 x 2pi m/s) and its heading was perturbed by Gaussian angular noise (sigma = 5.76 x 2 rad/s). A wall-avoidance mechanism slowed the agent and biased turning when within 0.03 m of arena boundaries (slowdown factor = 0.25). These parameters produced smooth, autocorrelated trajectories resembling rodent foraging behavior.

To investigate how trajectory predictability influences the emergence of predictive representations, we implemented three trajectory regimes:
- **Random walk** (default): Smooth, correlated motion with gradual drift and turns, mimicking natural foraging.
- **Straight**: Fixed heading and speed (e.g., 0.8 m/s), producing highly predictable linear paths.
- **Per-step random**: Heading and speed resampled independently at every time step, producing maximally unpredictable motion.

Additional trajectory control parameters included velocity smoothing (exponential moving average factor), speed scaling, maximum speed caps, and rotational noise scaling, enabling systematic manipulation of trajectory predictability along a continuum.

## Grid Score Computation

Spatial tuning of RNN hidden units was quantified using established grid score metrics (Sargolini et al., 2006; Langston et al., 2010). For each unit, a rate map R_u(x, y) was constructed by binning the arena into a 20 x 20 grid and computing the mean activation in each spatial bin across concatenated trajectories. The two-dimensional spatial autocorrelogram (SAC) was then computed via the Pearson correlation between the rate map and spatially shifted copies of itself.

Gridness was assessed by correlating the SAC with rotated versions of itself at angles of 30, 45, 60, 90, 120, 135, and 150 degrees. The 60-degree grid score was defined as:

grid_60 = (rho_60 + rho_120) / 2 - (rho_30 + rho_90 + rho_150) / 3

where rho_theta is the Pearson correlation between the SAC and its rotation by theta degrees within an annular mask. The 90-degree grid score was computed analogously:

grid_90 = rho_90 - (rho_45 + rho_135) / 2

To ensure robustness, scores were computed across 10 concentric annular masks with inner radius fixed at 0.2 and outer radius varying from 0.4 to 1.0 (in units of the SAC half-width). The maximum score across all masks was retained for each unit.

## Temporal Shift Analysis for Predictive and Retrospective Coding

To identify units encoding future or past spatial positions, we employed a temporal shift analysis inspired by Ouchi and Fujisawa (2024). For a given integer lag k, activations at time t were aligned with positions at time t + k:

D_k = {(p_{t+k}, g_{t,u}) | 0 <= t < T - |k|}

Positive lags (k > 0) test whether a unit's current activity predicts the agent's future position, while negative lags (k < 0) assess retrospective (postdictive) coding. For each lag, a shifted rate map was computed and scored for gridness using the procedure described above. This yielded a gridness-versus-lag profile s_60(k, u) for each unit u, evaluated over lags k in [-20, +20] time steps.

Integer lags were converted to physical distances (cm) using the mean displacement per time step:

Delta_cm = 100 x (1 / (T - 1)) * sum_{t=0}^{T-2} ||p_{t+1} - p_t||_2

This conversion ensured that all results are reported in biologically interpretable spatial units rather than abstract step counts.

## Cell Classification

Hidden units were classified into functional categories based on their gridness-versus-lag profiles. A unit was considered to exhibit significant spatial tuning if its peak gridness score (across all lags) exceeded a threshold of 0.2. Among spatially tuned units, we defined three classes:

- **Predictive grid cells**: Units whose peak gridness occurred at a positive lag corresponding to a spatial shift >= 5 cm (i.e., the unit's firing field is maximally grid-like when aligned with future positions).
- **Retrospective grid cells**: Units whose peak gridness occurred at a negative lag corresponding to a spatial shift <= -5 cm (i.e., aligned with past positions).
- **Zero-lag (normal) grid cells**: Units whose peak gridness occurred within +/- 5 cm of the current position.
- **Non-grid units**: Units whose peak gridness fell below the threshold at all lags.

For more stringent classification (used in select analyses), additional criteria were applied: the zero-lag gridness was required to exceed 0.5, and the peak shifted gridness was independently required to exceed 0.5.

### Shuffle Control

To establish statistical significance of temporal coding, we implemented a shuffle control procedure. For each unit, the activation time series was randomly permuted 100 times, and gridness scores were recomputed for each shuffle at each lag. A unit's observed gridness at a given lag was considered significant only if it exceeded the 95th percentile of the shuffle distribution (alpha = 0.05). This ensured that classifications were not driven by chance correlations between spatial binning and activation patterns.

## Preferred Shift Distribution Analysis

For predictive and retrospective grid cells, we quantified the distribution of preferred temporal shifts (the lag at which each unit achieved its maximum gridness). Across all five seeds, we computed the following breadth metrics: standard deviation (SD), interquartile range (IQR), 10th--90th percentile span, and coefficient of variation (CV). These metrics were computed both per-seed and on a pooled distribution concatenated across all seeds, characterizing the diversity of predictive horizons within the network.

## Ablation Studies

To establish the causal contribution of each cell class to path integration performance, we performed systematic ablation experiments. Ablation was implemented by zeroing the weights (encoder, decoder, and recurrent connections) associated with selected hidden units, effectively silencing them. Performance was assessed as the mean decoding error (in cm) on a held-out set of 8 evaluation trajectory batches.

### Class-Specific Ablations

We tested all eight possible combinations of class ablations: baseline (no ablation), predictive only, retrospective only, normal only, predictive + retrospective, predictive + normal, retrospective + normal, and all three classes combined. Within each class, units were ranked by their peak gridness score, and ablations were performed at multiple percentile thresholds (0, 5, 10, 15, 20, 25, 50, 75, and 100% of the class), enabling dose-response characterization.

### Random Ablation Controls

To control for nonspecific effects of unit removal, each targeted ablation was compared against matched random ablations. For each condition, the same number of units was randomly selected from the full population (excluding the targeted class) and ablated. This was repeated 3 times per condition, and the mean random-ablation error was reported alongside the targeted-ablation error. The difference between targeted and matched-random ablation effects quantified the specific functional importance of each cell class beyond what would be expected from removing an equivalent number of arbitrary units.

### Fixed-Count Ablations

In addition to percentile-based ablations, we performed fixed-count ablations (e.g., removing the top 25 or 50 units per class) to enable direct comparison of class importance independent of class size.

## Toroidal Structure Analysis

To investigate the geometric organization of population activity, we analyzed whether the network's representations exhibit toroidal topology, as observed in biological grid cell populations (Gardner et al., 2022).

### Lattice Vector Estimation

The fundamental lattice vectors (k1, k2) of the grid cell population were estimated from the average rate map across grid-classified units. A two-dimensional Fast Fourier Transform (FFT) was applied to the average rate map, and the dominant frequency peak (excluding DC) was identified as k1. The second lattice vector k2 was computed at a 60-degree rotation from k1, consistent with the hexagonal symmetry of grid cell firing patterns.

### Phase-Based Torus Embedding

Each unit's position on the torus was determined by computing the complex phase of its rate map at the two fundamental frequencies:

phase_j(u) = arg(sum_{x,y} R_u(x,y) * exp(-2*pi*i * k_j . (x, y)))

for j in {1, 2}. These phases defined a two-dimensional toroidal coordinate for each unit, with theta_1 and theta_2 representing the angular positions on the major and minor torus axes, respectively.

Hidden state trajectories were projected onto the torus by computing the amplitude and phase of the population activity vector along each lattice direction. The resulting coordinates (theta_1, theta_2, r_1, r_2) were embedded in three-dimensional space as:

x = (R + r * cos(theta_2)) * cos(theta_1)
y = (R + r * cos(theta_2)) * sin(theta_1)
z = r * sin(theta_2)

with major radius R = 1.0 and minor radius r = 0.35.

### Toroidal Cell Detection

To identify the subset of units underlying toroidal population dynamics, we employed a data-driven clustering approach. The spatial autocorrelogram of each unit was analyzed for rotational symmetry: the SAC was correlated against 36 rotated versions (0--180 degrees), producing a rotational profile. Features extracted from this profile included a band-like score (correlation with a cos(2*theta) template), a grid-like score (correlation with a cos(3*theta) template), and an anisotropy measure (peak power / mean power in FFT). These features were standardized, embedded into two dimensions using UMAP, and clustered with DBSCAN (epsilon calibrated at 1.2 times the median k-nearest-neighbor distance, minimum samples = 6). Clusters were ranked by their combined band and anisotropy scores, and up to three clusters with band_score_mean >= 0.15, size >= 4 units, and angular separation >= 20 degrees were selected. The union of selected clusters defined the toroidal cell ensemble.

### Phase-to-Position Decoding

To quantify how faithfully the toroidal manifold tracks the agent's position, we decoded position from phase trajectories. Phase was unwrapped across time, and position differences were estimated via:

Delta_pos = Delta_theta * K_inv^T

where K_inv is the inverse of the lattice vector matrix. The root-mean-square error (RMSE) between decoded and true positions was computed both step-wise and cumulatively.

### Manifold Stability Under Ablation

To test whether predictive grid cells contribute to maintaining activity on the toroidal manifold, we measured the coefficient of variation (CV) of the toroidal radii r_1 and r_2 across time steps. Lower CV indicates tighter adherence to the manifold. We compared radius CV for the intact network against networks with predictive, retrospective, normal, and toroidal grid cells ablated. Additionally, we tracked the off-manifold distance: hidden states from ablated networks were compared against a baseline point cloud (80,000 reference states from the intact model) using k-nearest-neighbor distances (k = 5) in standardized activation space.

## Off-Manifold Distance Analysis

To further quantify the impact of ablations on population dynamics, we constructed a reference manifold from baseline network activity. One thousand trajectories were generated with the intact model, producing a cloud of hidden states (capped at 80,000 points). States were z-scored and optionally projected via PCA. For each ablation condition (predictive, retrospective, random controls) at multiple percentile thresholds (25%, 75%, 100%), 100 evaluation trajectories were run through the ablated model, and the mean k-nearest-neighbor distance (k = 5) to the baseline cloud was computed at each time step. This off-manifold distance metric captures how much ablation disrupts the network's learned attractor dynamics.

## Border Cell Analysis

Border cells were identified using an established border score metric (Solstad et al., 2008). Connected firing fields were detected by thresholding each rate map at 30% of its peak value and retaining components with area > 200 cm^2. The border score was defined as:

border_score = (CM - DM) / (CM + DM)

where CM is the maximum fractional coverage of any single field along any wall, and DM is the mean firing-weighted distance to the nearest wall (normalized by half the arena width). Units with border_score >= 0.5 were classified as border cells.

### Predictive Border Cells

To test whether border cells also exhibit predictive coding, we computed border scores across the same temporal lags used for grid score analysis (k in [-20, +20] steps). For each unit, the lag yielding the maximum border score was identified and converted to centimeters. Predictive border cells were defined as those with best_shift_cm >= 5 cm and peak border_score >= 0.5; retrospective border cells had best_shift_cm <= -5 cm. The overlap between border cell classes and grid cell classes was quantified.

## Band Cell Analysis

Band cells were identified following the definition of Schaeffer et al. (2024). For each unit's rate map, the band score was computed as the maximum Pearson correlation between the (mean-centered, variance-normalized) rate map and a library of two-dimensional sinusoidal templates:

band_score(u) = max_{kx, ky in K} corr(R_u, cos(2*pi*(kx*X + ky*Y)))

where K spans spatial frequencies from 0.0 to 2.0 cycles/m in increments of 0.1. Units were classified as band cells if their score exceeded the 90th percentile of the population distribution (or an absolute threshold of 0.3--0.5). The overlap between band cells and predictive/retrospective/normal grid cell classes was quantified via cross-tabulation.

## Training Emergence Analysis

To characterize when and how predictive representations develop during learning, we analyzed intermediate training checkpoints saved at each epoch. For each checkpoint, we computed gridness scores at lag = 0 (standard grid cells) and across positive/negative lags (predictive/retrospective cells), applying the same classification criteria described above but with a gridness threshold of 0.3. We tracked the following metrics across training:

- **Counts and fractions**: The number and proportion of grid, predictive, retrospective, and predictive-intersect-grid cells at each epoch.
- **Mean gridness**: The average gridness score within each class, testing whether cells become "griddier" even if their count remains stable.
- **Mean preferred shift**: The average spatial shift (in cm) for predictive and retrospective cells, assessing whether predictive horizons change during learning.

To ensure fair comparison across epochs, the same set of cached evaluation trajectories (seed-controlled) was reused at each checkpoint. Training loss and position error curves were overlaid with emergence fractions to identify the temporal relationship between task acquisition and representational structure.

## Cross-Seed Summary Statistics

All analyses were aggregated across the five independently trained networks to assess generality. For cell class proportions, we report the mean +/- SEM fraction of units in each class. Preferred shift distributions were pooled across seeds to characterize the population-level diversity of predictive horizons. Summary figures include stacked bar charts of class fractions per seed, line plots of class counts across seeds, pooled histograms of preferred shifts, and mean +/- SEM bar plots with error bars.

## Trajectory Predictability Experiments

To test whether the emergence of predictive grid cells depends on trajectory statistics, we trained networks under varying motion regimes. Using the trajectory style parameter, we compared networks trained on highly predictable trajectories (straight paths with fixed speed, low rotational noise sigma_scale = 0.1, velocity smoothing = 0.2) against maximally unpredictable trajectories (per-step random heading and speed, no smoothing, high rotational noise sigma_scale = 2.0). Per-epoch grid diagnostics tracked whether predictive cell fractions and mean gridness differed between regimes, testing the hypothesis that predictive coding emerges preferentially when the agent's trajectory contains exploitable temporal structure (a "rich" learning regime) versus memorization-based solutions (a "lazy" regime).

## Torus Trajectory Variation Metric

To provide a single summary statistic capturing the impact of ablations on toroidal dynamics, we defined a combined variation metric:

combined = alpha * manifold_variation + (1 - alpha) * trajectory_variation

where alpha = 0.5. The manifold variation term quantifies the relative change in radius CV between baseline and ablated conditions, averaged across ablation types. The trajectory variation term captures how ablation effects differ across trajectory styles (random walk, straight, per-step random), combining the range-to-mean ratio of impacts with the within-style coefficient of variation of impacts across ablation percentages. This metric integrates manifold stability and trajectory sensitivity into a single interpretable score.

## Software and Reproducibility

All analyses were implemented in Python using PyTorch (model training and inference), NumPy and SciPy (grid score computation, spatial autocorrelation), scikit-learn (UMAP embedding, DBSCAN clustering, PCA), and Matplotlib/Seaborn (visualization). Model checkpoints, analysis outputs, and configuration parameters are available in the project repository. Random seeds were fixed for trajectory generation and model initialization to ensure reproducibility across runs.
