import math
import warnings

import matplotlib.pyplot as plt
import tensorflow as tf

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tensordrug.core.core import Registry as R
from tensordrug.data.graph import Graph, PackedGraph
from tensordrug.utils import decorator
from tensordrug.data.rdkit import draw


class Molecule(Graph):
    """
    Molecule graph with chemical features.
    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        formal_charge (array_like, optional): formal charges of shape :math:`(|V|,)`
        explicit_hs (array_like, optional): number of explicit hydrogens of shape :math:`(|V|,)`
        chiral_tag (array_like, optional): chirality tags of shape :math:`(|V|,)`
        radical_electrons (array_like, optional): number of radical electrons of shape :math:`(|V|,)`
        atom_map (array_likeb optional): atom mappings of shape :math:`(|V|,)`
        bond_stereo (array_like, optional): bond stereochem of shape :math:`(|E|,)`
        stereo_atoms (array_like, optional): ids of stereo atoms of shape :math:`(|E|,)`
    """

    bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
    atom2valence = {
        1: 1,
        5: 3,
        6: 4,
        7: 3,
        8: 2,
        9: 1,
        14: 4,
        15: 5,
        16: 6,
        17: 1,
        35: 1,
        53: 7,
    }
    bond2valence = [1, 2, 3, 1.5]
    id2bond = {v: k for k, v in bond2id.items()}
    empty_mol = Chem.MolFromSmiles("")
    dummy_mol = Chem.MolFromSmiles("CC")
    dummy_atom = dummy_mol.GetAtomWithIdx(0)
    dummy_bond = dummy_mol.GetBondWithIdx(0)

    def __init__(
        self,
        edge_list=None,
        atom_type=None,
        bond_type=None,
        formal_charge=None,
        explicit_hs=None,
        chiral_tag=None,
        radical_electrons=None,
        atom_map=None,
        bond_stereo=None,
        stereo_atoms=None,
        **kwargs
    ):
        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.bond2id)
        super(Molecule, self).__init__(edge_list=edge_list, **kwargs)
        atom_type, bond_type = self._standarize_atom_bond(atom_type, bond_type)

        formal_charge = self._standarize_attribute(formal_charge, self.num_node)
        explicit_hs = self._standarize_attribute(explicit_hs, self.num_node)
        chiral_tag = self._standarize_attribute(chiral_tag, self.num_node)
        radical_electrons = self._standarize_attribute(radical_electrons, self.num_node)
        atom_map = self._standarize_attribute(atom_map, self.num_node)
        bond_stereo = self._standarize_attribute(bond_stereo, self.num_edge)
        stereo_atoms = self._standarize_attribute(stereo_atoms, (self.num_edge, 2))

        with self.node():
            self.atom_type = atom_type
            self.formal_charge = formal_charge
            self.explicit_hs = explicit_hs
            self.chiral_tag = chiral_tag
            self.radical_electrons = radical_electrons
            self.atom_map = atom_map

        with self.edge():
            self.bond_type = bond_type
            self.bond_stereo = bond_stereo
            self.stereo_atoms = stereo_atoms

    def _standarize_atom_bond(self, atom_type, bond_type):
        if atom_type is None:
            raise ValueError("`atom_type` should be provided")
        if bond_type is None:
            raise ValueError("`bond_type` should be provided")

        atom_type = tf.convert_to_tensor(atom_type, dtype=tf.dtypes.int32)
        bond_type = tf.convert_to_tensor(bond_type, dtype=tf.dtypes.int32)
        return atom_type, bond_type

    def _standarize_attribute(self, attribute, size):
        if attribute is not None:
            attribute = tf.convert_to_tensor(attribute, dtype=tf.dtypes.int32)
        else:
            if isinstance(size, tf.Tensor):
                size = size.tolist()
            attribute = tf.zeros(size, dtype=tf.dtypes.int32)
        return attribute

    @classmethod
    def _standarize_option(cls, option):
        if option is None:
            option = []
        elif isinstance(option, str):
            option = [option]
        return option

    def _check_no_stereo(self):
        if (self.bond_stereo.numpy() > 0).any():
            warnings.warn(
                "Try to apply masks on molecules with stereo bonds. This may produce invalid molecules. "
                "To discard stereo information, call `mol.bond_stereo[:] = 0` before applying masks."
            )

    def _maybe_num_node(self, edge_list):
        if len(edge_list):
            return edge_list[:, :2].max().item() + 1
        else:
            return 0

    @classmethod
    def from_smiles(
        cls,
        smiles,
        node_feature="default",
        edge_feature="default",
        graph_feature=None,
        with_hydrogen=False,
        kekulize=False,
    ):
        """
        Create a molecule from a SMILES string.
        Parameters:
            smiles (str): SMILES string
            node_feature (str or list of str, optional): node features to extract
            edge_feature (str or list of str, optional): edge features to extract
            graph_feature (str or list of str, optional): graph features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES `%s`" % smiles)

        return cls.from_molecule(
            mol, node_feature, edge_feature, graph_feature, with_hydrogen, kekulize
        )

    @classmethod
    def from_molecule(
        cls,
        mol,
        node_feature="default",
        edge_feature="default",
        graph_feature=None,
        with_hydrogen=False,
        kekulize=False,
    ):
        """
        Create a molecule from a RDKit object.
        Parameters:
            mol (rdchem.Mol): molecule
            node_feature (str or list of str, optional): node features to extract
            edge_feature (str or list of str, optional): edge features to extract
            graph_feature (str or list of str, optional): graph features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if mol is None:
            mol = cls.empty_mol

        if with_hydrogen:
            mol = Chem.AddHs(mol)
        if kekulize:
            Chem.Kekulize(mol)

        node_feature = cls._standarize_option(node_feature)
        edge_feature = cls._standarize_option(edge_feature)
        graph_feature = cls._standarize_option(graph_feature)

        atom_type = []
        formal_charge = []
        explicit_hs = []
        chiral_tag = []
        radical_electrons = []
        atom_map = []
        _node_feature = []
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())] + [
            cls.dummy_atom
        ]
        for atom in atoms:
            atom_type.append(atom.GetAtomicNum())
            formal_charge.append(atom.GetFormalCharge())
            explicit_hs.append(atom.GetNumExplicitHs())
            chiral_tag.append(atom.GetChiralTag())
            radical_electrons.append(atom.GetNumRadicalElectrons())
            atom_map.append(atom.GetAtomMapNum())
            feature = []
            for name in node_feature:
                func = R.get("features.atom.%s" % name)
                feature += func(atom)
            _node_feature.append(feature)
        atom_type = tf.convert_to_tensor(atom_type)[:-1]
        atom_map = tf.convert_to_tensor(atom_map)[:-1]
        formal_charge = tf.convert_to_tensor(formal_charge)[:-1]
        explicit_hs = tf.convert_to_tensor(explicit_hs)[:-1]
        chiral_tag = tf.convert_to_tensor(chiral_tag)[:-1]
        radical_electrons = tf.convert_to_tensor(radical_electrons)[:-1]
        if len(node_feature) > 0:
            _node_feature = tf.convert_to_tensor(_node_feature)[:-1]
        else:
            _node_feature = None

        edge_list = []
        bond_type = []
        bond_stereo = []
        stereo_atoms = []
        _edge_feature = []
        bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())] + [
            cls.dummy_bond
        ]
        for bond in bonds:
            type = str(bond.GetBondType())
            stereo = bond.GetStereo()
            if stereo:
                _atoms = [a for a in bond.GetStereoAtoms()]
            else:
                _atoms = [0, 0]
            if type not in cls.bond2id:
                continue
            type = cls.bond2id[type]
            h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list += [[h, t, type], [t, h, type]]
            # always explicitly store aromatic bonds, no matter kekulize or not
            if bond.GetIsAromatic():
                type = cls.bond2id["AROMATIC"]
            bond_type += [type, type]
            bond_stereo += [stereo, stereo]
            stereo_atoms += [_atoms, _atoms]
            feature = []
            for name in edge_feature:
                func = R.get("features.bond.%s" % name)
                feature += func(bond)
            _edge_feature += [feature, feature]
        edge_list = edge_list[:-2]
        bond_type = tf.convert_to_tensor(bond_type)[:-2]
        bond_stereo = tf.convert_to_tensor(bond_stereo)[:-2]
        stereo_atoms = tf.convert_to_tensor(stereo_atoms)[:-2]
        if len(edge_feature) > 0:
            _edge_feature = tf.convert_to_tensor(_edge_feature)[:-2]
        else:
            _edge_feature = None

        _graph_feature = []
        for name in graph_feature:
            func = R.get("features.molecule.%s" % name)
            _graph_feature += func(mol)
        if len(graph_feature) > 0:
            _graph_feature = tf.convert_to_tensor(_graph_feature)
        else:
            _graph_feature = None

        num_relation = len(cls.bond2id) - 1 if kekulize else len(cls.bond2id)
        return cls(
            edge_list,
            atom_type,
            bond_type,
            formal_charge=formal_charge,
            explicit_hs=explicit_hs,
            chiral_tag=chiral_tag,
            radical_electrons=radical_electrons,
            atom_map=atom_map,
            bond_stereo=bond_stereo,
            stereo_atoms=stereo_atoms,
            node_feature=_node_feature,
            edge_feature=_edge_feature,
            graph_feature=_graph_feature,
            num_node=mol.GetNumAtoms(),
            num_relation=num_relation,
        )

    def to_smiles(self, isomeric=True, atom_map=True, canonical=False):
        """
        Return a SMILES string of this molecule.
        Parameters:
            isomeric (bool, optional): keep isomeric information or not
            atom_map (bool, optional): keep atom mapping or not
            canonical (bool, optional): if true, return the canonical form of smiles
        Returns:
            str
        """
        mol = self.to_molecule()
        if not atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
        if canonical:
            smiles_set = set()
            while smiles not in smiles_set:
                smiles_set.add(smiles)
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
        return smiles

    def to_molecule(self, ignore_error=False):
        """
        Return a RDKit object of this molecule.
        Parameters:
            ignore_error (bool, optional): if true, return ``None`` for illegal molecules.
                Otherwise, raise an exception.
        Returns:
            rdchem.Mol
        """
        mol = Chem.RWMol()
        atom_type = self.atom_type.numpy().tolist()
        bond_type = self.bond_type.numpy().tolist()
        formal_charge = self.formal_charge.numpy().tolist()
        explicit_hs = self.explicit_hs.numpy().tolist()
        chiral_tag = self.chiral_tag.numpy().tolist()
        radical_electrons = self.radical_electrons.numpy().tolist()
        atom_map = self.atom_map.numpy().tolist()
        bond_stereo = self.bond_stereo.numpy().tolist()
        stereo_atoms = self.stereo_atoms.numpy().tolist()
        for i in range(self.num_node):
            atom = Chem.Atom(atom_type[i])
            atom.SetFormalCharge(formal_charge[i])
            atom.SetNumExplicitHs(explicit_hs[i])
            atom.SetChiralTag(Chem.ChiralType(chiral_tag[i]))
            atom.SetNumRadicalElectrons(radical_electrons[i])
            atom.SetNoImplicit(explicit_hs[i] > 0 or radical_electrons[i] > 0)
            atom.SetAtomMapNum(atom_map[i])
            mol.AddAtom(atom)
        edge_list = self.edge_list.numpy().tolist()
        for i in range(self.num_edge):
            h, t, type = edge_list[i]
            if h < t:
                j = mol.AddBond(h, t, Chem.BondType.names[self.id2bond[type]])
                bond = mol.GetBondWithIdx(j - 1)
                bond.SetIsAromatic(bond_type[i] == self.bond2id["AROMATIC"])
                bond.SetStereo(Chem.BondStereo(bond_stereo[i]))
        j = 0
        for i in range(self.num_edge):
            h, t, type = edge_list[i]
            if h < t:
                if bond_stereo[i]:
                    bond = mol.GetBondWithIdx(j)
                    bond.SetStereoAtoms(*stereo_atoms[i])
                j += 1
        if ignore_error:
            try:
                with utils.no_rdkit_log():
                    mol.UpdatePropertyCache()
                    Chem.AssignStereochemistry(mol)
                    mol.ClearComputedProps()
                    mol.UpdatePropertyCache()
            except:
                mol = None
        else:
            mol.UpdatePropertyCache()
            Chem.AssignStereochemistry(mol)
            mol.ClearComputedProps()
            mol.UpdatePropertyCache()

        return mol

    def ion_to_molecule(self):
        """
        Convert ions to molecules by adjusting hydrogens and electrons.
        Note [N+] will not be converted.
        """
        data_dict = self.data_dict

        formal_charge = data_dict.pop("formal_charge")
        explicit_hs = data_dict.pop("explicit_hs")
        radical_electrons = data_dict.pop("radical_electrons")
        pos_nitrogen = (self.atom_type == 7) & (self.explicit_valence > 3)
        formal_charge = pos_nitrogen.long()
        explicit_hs = tf.zeros_like(explicit_hs)
        radical_electrons = tf.zeros_like(radical_electrons)

        return type(self)(
            self.edge_list,
            edge_weight=self.edge_weight,
            num_node=self.num_node,
            num_relation=self.num_relation,
            formal_charge=formal_charge,
            explicit_hs=explicit_hs,
            radical_electrons=radical_electrons,
            meta_dict=self.meta_dict,
            **data_dict
        )

    def to_scaffold(self, chirality=False):
        """
        Return a scaffold SMILES string of this molecule.
        Parameters:
            chirality (bool, optional): consider chirality in the scaffold or not
        Returns:
            str
        """
        smiles = self.to_smiles()
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles, includeChirality=chirality
        )
        return scaffold

    def node_mask(self, index, compact=False):
        self._check_no_stereo()
        return super(Molecule, self).node_mask(index, compact)

    def edge_mask(self, index):
        self._check_no_stereo()
        return super(Molecule, self).edge_mask(index)

    def undirected(self, add_inverse=False):
        if add_inverse:
            raise ValueError(
                "Bonds are undirected relations, but `add_inverse` is specified"
            )
        return super(Molecule, self).undirected(add_inverse)

    @property
    def num_atom(self):
        """Number of atoms."""
        return self.num_node

    @property
    def num_bond(self):
        """Number of bonds."""
        return self.num_edge

    @decorator.cached_property
    def explicit_valence(self):
        bond2valence = tf.convert_to_tensor(self.bond2valence)
        explicit_valence = scatter_add(
            bond2valence[self.edge_list[:, 2]],
            self.edge_list[:, 0],
            dim_size=self.num_node,
        )
        return explicit_valence.round().long()

    @decorator.cached_property
    def is_valid(self):
        """A coarse implementation of valence check."""
        # TODO: cross-check by any domain expert
        atom2valence = tf.convert_to_tensor(float("nan")).repeat(constant.NUM_ATOM)
        for k, v in self.atom2valence:
            atom2valence[k] = v
        atom2valence = tf.convert_to_tensor(atom2valence)

        max_atom_valence = atom2valence[self.atom_type]
        # special case for nitrogen
        pos_nitrogen = (self.atom_type == 7) & (self.formal_charge == 1)
        max_atom_valence[pos_nitrogen] = 4
        if tf.match.is_nan(max_atom_valence).any():
            index = tf.match.is_nan(max_atom_valence).nonzero()[0]
            raise ValueError(
                "Fail to check valence. Unknown atom type %d" % self.atom_type[index]
            )

        is_valid = (self.explicit_valence <= max_atom_valence).all()
        return is_valid

    @decorator.cached_property
    def is_valid_rdkit(self):
        try:
            with utils.no_rdkit_log():
                mol = self.to_molecule()
                Chem.SanitizeMol(
                    mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES
                )
            is_valid = tf.ones(1, dtype=tf.dtypes.bool)
        except ValueError:
            is_valid = tf.zeros(1, dtype=tf.dtypes.bool)
        return is_valid

    def visualize(
        self, title=None, save_file=None, figure_size=(3, 3), ax=None, atom_map=False
    ):
        """
        Visualize this molecule with matplotlib.
        Parameters:
            title (str, optional): title for this molecule
            save_file (str, optional): ``png`` or ``pdf`` file to save visualization.
                If not provided, show the figure in window.
            figure_size (tuple of int, optional): width and height of the figure
            ax (matplotlib.axes.Axes, optional): axis to plot the figure
            atom_map (bool, optional): visualize atom mapping or not
        """
        is_root = ax is None
        if ax is None:
            fig = plt.figure(figsize=figure_size)
            if title is not None:
                ax = plt.gca()
            else:
                ax = fig.add_axes([0, 0, 1, 1])
        if title is not None:
            ax.set_title(title)

        mol = self.to_molecule()
        if not atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        draw.MolToMPL(mol, ax=ax)
        ax.set_frame_on(False)

        if is_root:
            if save_file:
                fig.savefig(save_file)
            else:
                fig.show()
                plt.show(block=True)

    def __eq__(self, other):
        smiles = self.to_smiles(isomeric=False, atom_map=False, canonical=True)
        other_smiles = other.to_smiles(isomeric=False, atom_map=False, canonical=True)
        return smiles == other_smiles
