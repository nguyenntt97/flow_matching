"""Flow2BT: distilling the continuous flow teacher into a reactive Behavior Tree.

Subsystem map (see ``survey/design/``):

===========  ==========================  ==============================================
Subsystem    Module                      Role
===========  ==========================  ==============================================
1            (upstream)                  ``utils/navmesh.py`` + the ``control`` tensor
2            ``src/models/flow_teacher``  conditional flow-matching teacher
3            ``rollout``, ``clustering``  trajectory ensembles, bifurcation dendrogram
4            ``features``, ``conditions`` physical feature space, SVM guard hyperplanes
5            ``primitives``               DMP action leaves
6            ``bt``, ``assembly``         reactive tri-state Behavior Tree
7a           ``cbf``                      CBF-QP safety filter
7b           ``verification``, ``falsify`` SMV export; bounded falsification
===========  ==========================  ==============================================
"""
