import sys
sys.path.append("../..")

import typing as t

import numpy as np
from tqdm import tqdm

import perceptrons.multilayer as mlp
import perceptrons.convex as conv

import geometry.interval as interval
import geometry.norms as norms
import geometry.circle as circle

class ConvApproxError:
    """
        A suite for testing the divergence between the
        Tight Convex Approximation and the original MLP.
        The test is performed on randomized MLPs and is
        performed as follows.

        1. We create a sequence of MLPs, all applied on the
        same input shape.
        2. The sequence will have an *increasing* number of
        layers.
            1. At each step, we create a MLP with `current_num_layers`, an
        integer in the range `{0, ..., max_num_layers}`.
            2. Each of the `current_num_layers` will have a *randomly* selected
        number of neurons `num_neurons`, an integer in the range
        `{2, ..., max_width}`.
        3. The next MLP in the sequence will have
        `new_current_num_layers  = current_num_layers + layer_step`.

        The metrics are obtained by sampling `num_samples` random points on the
        domain `[-1, 1]^d`, where `d` the input_shape.
    """
    def __init__(
            self,
            in_shape:       t.Tuple[int]    = [28, 28],
            max_num_layers: int             = 5,
            layer_step:     int             = 2,
            max_width:      int             = 50,
            num_samples:    int             = 1000,
            num_classes:    int             = 10,
            seed:           int             = 0,
            rad:            float           = None
        ) -> None:
        """
            **Inputs:**
            1. `in_shape: Tuple[int]`, the input shape of the MLP sequence.
            Default value: `[28, 28]`, the input shape of the MNIST dataset.
            2. `max_num_layers: int`, the maximum number of layers, for each MLP
            in the sequence. Default value: `5`.
            3. `layer_step: int`, the incremental step for each subsequent MLP of
            the sequence. Default value: `2`.
            4. `max_width: int`, the maximum number of neurons of each layer. Note
            that each layer will have a random number of neurons in the range
            `{2, ..., max_width}`. Default value: `50`.
            5. `num_samples: int`, the number of samples to be drawn from the input
            domain `[-1, 1]^d`. The samples remain the same for each MLP in the
            sequence. Default value `1000`.
            6. `num_classes:int`, the number of output classes, essentially the
            dimension of the networks output space. Default values: `10`.
            7. `seed: int`, the seed of the Random Number Generator.
            Default value: `0`.
        """
        ## Parameters
        self.in_shape       = in_shape
        self.max_num_layers = max_num_layers
        self.layer_step     = layer_step
        self.max_width      = max_width
        self.num_samples    = num_samples
        self.num_classes    = num_classes
        self.seed           = seed
        self.rad            = rad

        self.in_dim         = np.prod(self.in_shape)

        ## Interval
        self.input_domain   = interval.Interval(
                                -1.0 * np.ones(self.in_shape),
                                 1.0 * np.ones(self.in_shape)
                            )
        
        ## Generate Random Samples
        self.Samples = self.input_domain.random_points(
                            self.num_samples,
                            self.seed
                        )
        
        ## Create Architectures
        self.architectures  = []
        np.random.seed(self.seed)
        for num_layers in range(self.layer_step, self.max_num_layers, self.layer_step):
            architecture = [self.in_dim] +\
                list(np.random.randint(3, self.max_width, num_layers))
            
            # we append the last layer consisting of num_classes neurons
            architecture.append(self.num_classes)

            ## append new architecture to architectures
            self.architectures.append(architecture)


        
        ## Create MLPs and their Approximations
        self.relu_mlps  = []
        self.conv_mlps  = []
        ind             = 0
        for architecture in tqdm(self.architectures):
            relu_mlp = mlp.FromRandom(
                        architecture,
                        self.in_shape,
                        self.seed,
                        "mlp-" + str(ind)
                    )
            
            conv_mlp = None
            if self.rad is not None:
                B_rad = circle.InfCircle(
                    np.zeros(self.in_shape),
                    self.rad
                ).get_interval()
                conv_mlp = conv.ConvexApprox(
                    relu_mlp,
                    B_rad
                )
            else:
                conv_mlp = conv.ConvexApprox(
                    relu_mlp,
                    self.input_domain
                )

            self.relu_mlps.append(relu_mlp)
            self.conv_mlps.append(conv_mlp)
            ind += 1
        self.num_mlps = ind


        ## Compute Scores on Samples
        self.relu_scores = []
        self.conv_scores = []
        for i in tqdm(range(self.num_mlps)):
            self.relu_scores.append(self.relu_mlps[i].scores(self.Samples))
            self.conv_scores.append(self.conv_mlps[i].scores(self.Samples))
        

        ## Statistics
        # Divergemce between ReLU and Conv. Scores
        # absolute values
        self.diffs_vecs                         = []
        self.avg_tight_divergences              = []

        # normalized values for diff in diffs
        # diff_norm <-- diff / ||diff||_{+oo}
        self.diffs_vecs_norm                    = []
        self.avg_tight_divergences_norm         = []


        # Worst Case Lower Bounds
        # absolute values
        self.worst_lb_vecs                      = []
        self.worst_lb_tight_divergences         = []

        # normalized values for lb_vec in lb_vecs
        # lb_vec_norm <-- lb_vec / ||lb_vec||_{+oo}
        self.worst_lb_vecs_norm                 = []
        self.worst_lb_tight_divergences_norm    = []


        # Worst Case Upper Bounds
        # absolute values
        self.worst_ub_vecs                      = []
        self.worst_ub_tight_divergences         = []
        
        # normalized values for ub_vec in ub_vecs
        # ub_vec_norm <-- ub_vec / ||ub_vec||_{+oo}
        self.worst_ub_vecs_norm                 = []
        self.worst_ub_tight_divergences_norm    = []
        

        ## States
        self.avg_tight_divergence_computed          = False
        self.worst_lb_tight_divergence_computed     = False
        self.worst_ub_tight_divergence_computed     = False
        

    ##########
    # Report #
    ##########

    def __str__(self):
        s =    "================================================================="  + "\n"
        s +=     "~~ Reporting ~~"                                                  + "\n"
        s +=    "=================================================================" + "\n"
        s +=    "Parameters:"                                                       + "\n"
        s +=    "_________________________________________________________________" + "\n"
        s +=    f"{'':<4}{'In. Shape.:':<19}{str(self.in_shape)}"                   + "\n"
        s +=    f"{'':<4}{'Num. Classes (Out. Dim.):':<19}{str(self.num_classes)}"  + "\n"
        s +=    f"{'':<4}{'Layer Step:':<19}{str(self.layer_step)}"                 + "\n"
        s +=    f"{'':<4}{'Max. Width:':<19}{str(self.max_width)}"                  + "\n"
        s +=    f"{'':<4}{'Num. Samples:':<19}{str(self.num_samples)}"              + "\n"
        s +=    f"{'':<4}{'Seed:':<19}{str(self.seed)}"                             + "\n"
        s +=    f"{'':<4}{'Num. Networks:':<19}{str(self.max_width)}"               + "\n"
        s +=    "=================================================================" + "\n\n\n"
        
        return s
    
    def report(self) -> None:
        print(self)
    
    def report_architectures(self) -> None:
        s =    "================================================================="  + "\n"
        s +=     "~~ Architectures ~~"                                              + "\n"
        s +=    "=================================================================" + "\n"
        ind = 1
        for architecture in self.architectures:
            s += f"{str(ind) + ')':<4}" + str(architecture) + "\n"
            ind += 1
        
        print(s)


    ###################
    # Compute Metrics #
    ###################
    def comp_avg_tight_divergence(self) -> t.List[float]:
        """
            Computing the metric:
            ```
                AvgErr  = (1/|D|) * Sum_{x \in D} err(x)
                        = (1/|D|) * Sum_{x \in D} ||z(x) - s(x||_{+oo},
            ```
            where `D` is the set of random samples drawn from the input
            domain `[-1, 1]^d`. Above `z` is output of the original MLP,
            and `s` is the output of the Tight Convex Approximation.
        """
        if self.avg_tight_divergence_computed: return self.avg_tight_divergences

        for i in tqdm(range(self.num_mlps)):
            # compute diffs
            diff_vec = self.relu_scores[i] - self.conv_scores[i]
            self.diffs_vecs.append(diff_vec)
            
            # compute tight divergence
            #tight_divergence       = np.max(np.abs(diff_vec), 0)
            tight_divergence       = norms.inf_norm(diff_vec)
            
            # compute new average
            avg_tight_divergence   = np.sum(tight_divergence) / len(tight_divergence)
            
            # append average
            self.avg_tight_divergences.append(avg_tight_divergence)
        
        self.avg_tight_divergence_computed = True
        return self.avg_tight_divergences



    def comp_worst_lb_tight_divergence(self) -> t.List[float]:
        """
            Computing the *lower bound* for the Worst Case Tight Divergence.
            This lower bound is taken as difference,
            ```
                WorstCaseLB = ||z(0) - s(0)||_{+oo}.
            ```
            Namely, the tight divergence on the all zero input.
        """
        if self.worst_lb_tight_divergence_computed: return self.worst_lb_tight_divergences

        #self.worst_lb_tight_divergences = []
        for ind in tqdm(range(self.num_mlps)):
            relu_score  = self.relu_mlps[ind].scores(np.zeros(self.in_shape))
            conv_score  = self.conv_mlps[ind].scores(np.zeros(self.in_shape))
            diff        = relu_score - conv_score

            worst_lb_tight_divergence = np.max(np.abs(diff))
            #worst_lb_tight_divergence = conv_score
            self.worst_lb_tight_divergences.append(worst_lb_tight_divergence)
        
        self.worst_lb_tight_divergence_computed = True
        return self.worst_lb_tight_divergences



    def comp_worst_ub_tight_divergence(self) -> t.List[float]:
        """
            Computing the *upper bound* for the Worst Case Tight Divergence.
            ```
                WorstCaseUB = ||s_max||_{+oo},
            ```
            where `s_max` is a value bounding the maximum value of the Tight
            Convex Approximation, i.e. for each `x`, `s(x) <= s_max`, where `<=` is
            taken coordinate-wise. In particular, `s_max` is the right endpoint of
            the interval I, which is the output bound of `s()` computed by the
            `bound_propagation` method.
        """
        if self.worst_ub_tight_divergence_computed: return self.worst_ub_tight_divergences

        #self.worst_ub_tight_divergences = []
        for ind in tqdm(range(self.num_mlps)):
            conv_range_ub = self.conv_mlps[ind].propagate_bounds(self.input_domain)[0][-1].ub
            #print(conv_range_ub)
            worst_ub_tight_divergence = np.max(np.abs(conv_range_ub))
            #print(worst_ub_tight_divergence)
            self.worst_ub_tight_divergences.append(worst_ub_tight_divergence)
        
        self.worst_ub_tight_divergence_computed = True
        return self.worst_ub_tight_divergences


    # def comp_avg_misclassification(self) -> t.List[float]:
    #     """
    #         Computing the metric:
    #         ```
    #             AvgMisClass = Sum_{x \in D} 1I[z(x) != s(x)]
    #         ```
    #         where `D` is the set of random samples drawn from the input
    #         domain `[-1, 1]^d`. Above `z` is output of the original MLP,
    #         and `s` is the output of the Tight Convex Approximation.
    #     """
    #     # if self.avg_misclassification_computed: return self.avg_misclassifications

    #     self.relu_preds = np.argmax(self.relu_scores, axis=2)
    #     self.conv_preds = np.argmax(self.conv_scores, axis=2)

    #     for i in tqdm(range(self.num_mlps)):
    #         misclassification      =\
    #             np.array(self.relu_preds[i]) != np.array(self.conv_preds[i])
            
    #         avg_misclassification  = np.sum(misclassification) / len(misclassification)
            
    #         self.avg_misclassifications.append(avg_misclassification)
        
    #     self.avg_misclassification = True
    #     return self.avg_misclassifications

    # def comp_avg_misclassification_class(self) ->np.ndarray:
    #     if self.avg_misclassification_class_computed: return self.avg_misclassification_class

    #     for i in tqdm(range(self.num_mlps)):
    #         for c in range(0, self.num_classes):
    #             # get the indices where the relu network predicts c
    #             c_inds              = np.argwhere(self.relu_preds[i] == c).flatten()
    #             misclass_class      = self.conv_preds[c_inds] != c
    #             avg_misclass_class  = np.sum(misclass_class) / len(c_inds)
                
    #             self.avg_misclassification_class[i, c] = avg_misclass_class
                
    #     self.avg_misclassification_class_computed = True
    #     return self.avg_misclassification_class
    
    # #################
    # # Sanity Checks #
    # #################,
    # def comp_relu_class_distribution(self) -> np.ndarray:
    #     if self.class_relu_distribution_computed: return self.class_relu_distribution

    #     for i in tqdm(range(self.num_mlps)):
    #         for c in range(0, self.num_classes):
    #             self.class_relu_distribution[i, c] =\
    #                 np.sum(self.relu_preds[i] == c) / len(self.relu_preds[i])
                
    #     self.class_relu_distribution_computed = True
    #     return self.class_relu_distribution
    

    # def comp_conv_class_distribution(self) -> np.ndarray:
    #     if self.class_conv_distribution_computed: return self.class_conv_distribution

    #     for i in tqdm(range(self.num_mlps)):
    #         for c in range(0, self.num_classes):
    #             self.class_conv_distribution[i, c] =\
    #                 np.sum(self.conv_preds[i] == c) / len(self.conv_preds[i])
                
    #     self.class_conv_distribution_computed = True
    #     return self.class_conv_distribution