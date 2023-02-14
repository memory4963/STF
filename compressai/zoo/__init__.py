# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from compressai.models import CCResRepWithoutY, CCEnhancedResRep, CC_manual_prune, CCResRepPruned, CCResRep, CC_RandomSplit, CC_uneven, CC_GD, SymmetricalTransFormer, WACNN, TransformerBasedCoding, DYSTF, CC, CCMainPrune, CCResRepMainPruned

from .pretrained import load_pretrained as load_state_dict

models = {
    'stf': SymmetricalTransFormer,
    'cnn': WACNN,
    'tbc': TransformerBasedCoding,
    'dystf': DYSTF,
    'cc': CC,
    'cc_resrep_main': CCMainPrune,
    'cc_resrep_main_pruned': CCResRepMainPruned,
    'cc_uneven': CC_uneven,
    'cc_gd': CC_GD,
    'cc_rand_split': CC_RandomSplit,
    'cc_resrep': CCResRep,
    'cc_resrep_pruned': CCResRepPruned,
    'cc_manual_prune': CC_manual_prune,
    'cc_enhanced_resrep': CCEnhancedResRep,
    'cc_resrep_without_y': CCResRepWithoutY,
}
