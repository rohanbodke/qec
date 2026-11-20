#!/usr/bin/env python3
"""
Exact construction and verification of Balasubramanian-Davydova-Lin (arXiv:2605.10943)
3D Self-Correcting Quantum Sheaf Codes Q_{(k)}(L, r) at arbitrary subdivision scale L and rib count r.
"""

from collections import defaultdict
import itertools


class CellComplex2D:
    """
    2D square cell complex X with sheaf data F on edges:
      - vertices: 0 .. V-1
      - edges: list of (u, v) with u < v, plus edge_color[e] in {'none', 'blue', 'red'}
        ('none' is used only when face-degree == 2, where blue and red agree: {00, 11})
      - faces: list of 4-tuples of vertices (v0, v1, v2, v3) in cyclic order around the square,
        or more generally (bottom-left, bottom-right, top-right, top-left) on local grid.
    """
    def __init__(self, num_vertices, edges, edge_colors, faces):
        self.num_vertices = num_vertices
        self.edges = edges                  # list of (u, v), u != v
        self.edge_colors = edge_colors      # list of 'blue', 'red', 'none'
        self.faces = faces                  # list of (v0, v1, v2, v3) in cyclic order

        # Map canonical undirected vertex pair -> list of edge indices
        # Note: in a regular cell complex without multi-edges, (min(u,v), max(u,v)) is unique!
        # Let's verify whether two edges ever share the exact same two endpoints.
        self.pair_to_edge = {}
        for idx, (u, v) in enumerate(edges):
            key = (min(u, v), max(u, v))
            if key in self.pair_to_edge:
                raise ValueError(f"Multi-edge detected on {key}!")
            self.pair_to_edge[key] = idx

        # Check all face boundary edges exist
        self.face_edges = []
        self.edge_faces = [[] for _ in range(len(edges))]
        self.vert_faces = [[] for _ in range(num_vertices)]
        self.vert_edges = [[] for _ in range(num_vertices)]

        for e_idx, (u, v) in enumerate(edges):
            self.vert_edges[u].append(e_idx)
            self.vert_edges[v].append(e_idx)

        for f_idx, (v0, v1, v2, v3) in enumerate(faces):
            cyc = [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]
            fe = []
            for a, b in cyc:
                key = (min(a, b), max(a, b))
                if key not in self.pair_to_edge:
                    raise ValueError(f"Face {f_idx} missing edge {key}")
                e_idx = self.pair_to_edge[key]
                fe.append(e_idx)
                self.edge_faces[e_idx].append(f_idx)
            self.face_edges.append(fe)
            for v in (v0, v1, v2, v3):
                self.vert_faces[v].append(f_idx)


def make_seed_single_square():
    """
    BDL Section 6.2.2 Base case Q_{(0)}:
      - Single square: vertices 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1)
      - Horizontal edges (0,1) and (3,2) colored 'blue'
      - Vertical edges (0,3) and (1,2) colored 'red'
    """
    edges = [
        (0, 1),  # bottom horizontal -> blue
        (2, 3),  # top horizontal    -> blue
        (0, 3),  # left vertical     -> red
        (1, 2),  # right vertical    -> red
    ]
    colors = ['blue', 'blue', 'red', 'red']
    faces = [(0, 1, 2, 3)]
    return CellComplex2D(4, edges, colors, faces)


def make_seed_surface_code(dx, dz):
    """
    d_x x d_z planar surface code sheaf complex:
      Grid of dx x dz squares: (ix, iz) for 0 <= ix < dx, 0 <= iz < dz.
      Top & bottom boundary edges (iz=0 and iz=dz): 'blue'
      Left & right boundary edges (ix=0 and ix=dx): 'red'
      Interior edges (face-degree 2): 'none'
    Let's check: for dx=1, dz=1 this is the single square [[2,1,1]].
    For dx=2, dz=2: 4 faces, interior has 4 edges + 2*2 blue boundary edges = 8 edges? Wait:
    let's check exact [[n, k, (d_X, d_Z)]] of make_seed_surface_code(dx, dz)!
    """
    def vid(ix, iz):
        return iz * (dx + 1) + ix

    num_v = (dx + 1) * (dz + 1)
    edges = []
    colors = []
    # Horizontal edges: (ix, iz) -- (ix+1, iz)
    for iz in range(dz + 1):
        for ix in range(dx):
            u, v = vid(ix, iz), vid(ix + 1, iz)
            edges.append((u, v))
            if iz == 0 or iz == dz:
                colors.append('blue')
            else:
                colors.append('none')
    # Vertical edges: (ix, iz) -- (ix, iz+1)
    for iz in range(dz):
        for ix in range(dx + 1):
            u, v = vid(ix, iz), vid(ix, iz + 1)
            edges.append((u, v))
            if ix == 0 or ix == dx:
                colors.append('red')
            else:
                colors.append('none')

    faces = []
    for iz in range(dz):
        for ix in range(dx):
            faces.append((vid(ix, iz), vid(ix + 1, iz), vid(ix + 1, iz + 1), vid(ix, iz + 1)))

    return CellComplex2D(num_v, edges, colors, faces)


def apply_iteration(complex_in: CellComplex2D, L: int, rib_coords: list, iter_type: str) -> CellComplex2D:
    """
    Apply one topological refinement + doubling step R_X (iter_type='X', new color='blue')
    or R_Z (iter_type='Z', new color='red') with subdivision factor L and interior rib positions
    rib_coords subset of {1, ..., L-1}.

    Let's check every detail against BDL Section 6.2:
      1. Refinement X -> X':
         - Each original vertex v stays a vertex.
         - Each original edge e=(u, w) (canonical orientation min->max) is subdivided into L segments
           with L-1 interior vertices (e, 1), ..., (e, L-1).
           Inherited edge color: if deg_face(e) != 2, all L segments inherit edge_color[e]; else 'none'.
         - Each original face f=(v0, v1, v2, v3) is parameterized on {0..L} x {0..L}:
           (0,0)=v0, (L,0)=v1, (L,L)=v2, (0,L)=v3.
           Boundary grid points (s, 0), (L, t), (s, L), (0, t) are identified with the subdivided
           vertices of the 4 boundary edges of f (respecting orientation!).
           Interior grid points (s, t) for 1 <= s, t <= L-1 are new vertices (f, s, t).
           Subdivided squares: for 0 <= s, t < L, square ((s,t), (s+1,t), (s+1,t+1), (s,t+1)).
         - What are the attaching ribs R subset X'(1)?
           In each face f, for r in rib_coords:
             - vertical rib segments: (r, t) -- (r, t+1) for 0 <= t < L
             - horizontal rib segments: (s, r) -- (s+1, r) for 0 <= s < L
           WAIT: check consistency across adjacent faces sharing an original edge e!
           Suppose face f has boundary edge e along t=0 (so s runs 0..L along e).
           At s = r in rib_coords, the rib in f hits the boundary vertex (e, r) PERPENDICULARLY!
           What if rib_coords is symmetric under s <-> L - s (i.e. r in rib_coords <=> L - r in rib_coords)?!
           HOLD ON!! CHECK ORIENTATION OF SHARED EDGES!!
           If two faces f1, f2 share an edge e, and e is parameterized in OPPOSITE directions on f1 and f2,
           then coordinate r from one end of e is coordinate L - r from the other end!!
           WAIT A SECOND!! Look at BDL:
             - For L=8, candidate ribs are {3, 4, 5} (symmetric under 8 - x: 8-3=5, 8-4=4!), and actual ribs are {4, 5}?
               Wait: in BDL's Z^3 embedding, ribs are defined by GLOBAL coordinates x, y, z mod ell, OR if canonical edge orientation min(u,v)->max(u,v) is used on every edge?!
               WAIT: On a square f=(v0, v1, v2, v3), opposite edges (v0,v1) and (v3,v2) must have COMPATIBLE rib coordinates!
               Let's check! Can every square face in X_{(i)} be given a bipartite/consistent grid orientation so that opposite edges of every square agree in orientation?!
               YES!! Let's prove that by induction!!
               - In X_{(0)} (any grid surface code), every vertex v has integer coordinates (x_v, y_v, z_v) or simpler:
                 assign every square f a local coordinate (s, t) in [0,1]^2 where s increases along edge (v0->v1) AND (v3->v2),
                 and t increases along (v0->v3) AND (v1->v2)!
                 If EVERY edge e in X_{(i)} has a PREFERRED orientation u -> v such that on EVERY face f=(v0,v1,v2,v3),
                 the oriented edges are v0->v1, v3->v2, v0->v3, v1->v2:
                 Does doubling preserve this property?!
                 CHECK:
                 1. On X'_{(i)} copy 0 and copy 1: subdividing a consistently oriented square grid produces a consistently oriented L x L square grid!
                 2. On X_{cyl}: every rib edge e_rib in X'_{(i)} already has a preferred orientation a -> b!
                    Then the cylinder square above e_rib has bottom edge a_0 -> b_0, top edge a_1 -> b_1,
                    left vertical edge a_0 -> a_1, and right vertical edge b_0 -> b_1!!
                    Its opposite edges are (a_0 -> b_0, a_1 -> b_1) and (a_0 -> a_1, b_0 -> b_1) — 100% CONSISTENTLY ORIENTED!!
                 MIND BLOWN!! Every complex X_{(i)} is AUTOMATICALLY a consistently-oriented cubical complex for ALL i >= 0,
                 for ANY L >= 2 and ANY subset rib_coords subset {1, ..., L-1} (even asymmetric ones like {1} in L=2 or {1,2} in L=3)!!
    """
    assert iter_type in ('X', 'Z')
    new_sheaf_color = 'blue' if iter_type == 'X' else 'red'

    # First verify input faces are consistently oriented or orient edges consistently:
    # Wait: let's store faces as (v00, v10, v11, v01) where bottom=(v00->v10), top=(v01->v11), left=(v00->v01), right=(v10->v11)!
    # Note: in cyclic order around the square: v00 -> v10 -> v11 -> v01 -> v00!
    # So v0=v00, v1=v10, v2=v11, v3=v01 means:
    #   bottom: v0 -> v1
    #   top:    v3 -> v2
    #   left:   v0 -> v3
    #   right:  v1 -> v2
    # Let's check if our edges have consistent directions across all faces!
    edge_dir = {}  # edge_idx -> (tail, head)
    for f_idx, (v0, v1, v2, v3) in enumerate(complex_in.faces):
        directed_sides = [(v0, v1), (v3, v2), (v0, v3), (v1, v2)]
        for a, b in directed_sides:
            e_idx = complex_in.pair_to_edge[(min(a, b), max(a, b))]
            if e_idx in edge_dir:
                if edge_dir[e_idx] != (a, b):
                    raise ValueError(f"Orientation conflict on edge {e_idx} in face {f_idx}!")
            else:
                edge_dir[e_idx] = (a, b)

    # Step 1: Build X' (single refined complex)
    xp_verts = []
    def add_xp_vert():
        vid = len(xp_verts)
        xp_verts.append(vid)
        return vid

    # Original vertices in X'
    orig_v_map = {v: add_xp_vert() for v in range(complex_in.num_vertices)}

    # Subdivided edge vertices in X': along edge_dir[e_idx] = (a, b), points k=0..L
    edge_pts = {}
    xp_edges = {}       # (min, max) -> (u, v_dir, inherited_color, is_rib, orig_edge_or_None)
    # Wait: track directed edges in X' too so next level stays consistently oriented!

    for e_idx in range(len(complex_in.edges)):
        a, b = edge_dir[e_idx]
        pts = [orig_v_map[a]] + [add_xp_vert() for _ in range(L - 1)] + [orig_v_map[b]]
        edge_pts[e_idx] = pts
        fdeg = len(complex_in.edge_faces[e_idx])
        inh_col = complex_in.edge_colors[e_idx] if fdeg != 2 else 'none'
        for k in range(L):
            u, w = pts[k], pts[k + 1]
            key = (min(u, w), max(u, w))
            xp_edges[key] = {
                'dir': (u, w),
                'color': inh_col,
                'is_rib': False,
                'orig_edge': e_idx,
            }

    # Track which vertices of X' lie on ribs (for cylinder vertical edges)
    rib_set = set(rib_coords)
    xp_faces = []  # (v00, v10, v11, v01)

    for f_idx, (v0, v1, v2, v3) in enumerate(complex_in.faces):
        e_bot = complex_in.pair_to_edge[(min(v0, v1), max(v0, v1))]
        e_top = complex_in.pair_to_edge[(min(v3, v2), max(v3, v2))]
        e_lft = complex_in.pair_to_edge[(min(v0, v3), max(v0, v3))]
        e_rgt = complex_in.pair_to_edge[(min(v1, v2), max(v1, v2))]

        grid = [[None] * (L + 1) for _ in range(L + 1)]
        for s in range(L + 1):
            grid[s][0] = edge_pts[e_bot][s]
            grid[s][L] = edge_pts[e_top][s]
        for t in range(L + 1):
            grid[0][t] = edge_pts[e_lft][t]
            grid[L][t] = edge_pts[e_rgt][t]
        for s in range(1, L):
            for t in range(1, L):
                grid[s][t] = add_xp_vert()

        # Interior horizontal edges: (s, t) -> (s+1, t) for 1 <= t <= L-1
        for t in range(1, L):
            is_r = (t in rib_set)
            for s in range(L):
                u, w = grid[s][t], grid[s + 1][t]
                key = (min(u, w), max(u, w))
                xp_edges[key] = {
                    'dir': (u, w),
                    'color': 'none',
                    'is_rib': is_r,
                    'orig_edge': None,
                }

        # Interior vertical edges: (s, t) -> (s, t+1) for 1 <= s <= L-1
        for s in range(1, L):
            is_r = (s in rib_set)
            for t in range(L):
                u, w = grid[s][t], grid[s][t + 1]
                key = (min(u, w), max(u, w))
                xp_edges[key] = {
                    'dir': (u, w),
                    'color': 'none',
                    'is_rib': is_r,
                    'orig_edge': None,
                }

        # Subdivided faces
        for s in range(L):
            for t in range(L):
                xp_faces.append((grid[s][t], grid[s + 1][t], grid[s + 1][t + 1], grid[s][t + 1]))

    # Step 2: Doubling Y = (X' x {0}) U (X' x {1}) U (R x [0,1])
    num_xp_v = len(xp_verts)
    total_verts = 2 * num_xp_v

    def v_copy(v, c):
        return v + c * num_xp_v

    out_edges = []
    out_colors = []
    out_faces = []

    # Collect rib vertices in X'
    rib_verts = set()
    for key, info in xp_edges.items():
        if info['is_rib']:
            u, w = info['dir']
            rib_verts.add(u)
            rib_verts.add(w)

    # Add edges of Copy 0 and Copy 1
    for key, info in xp_edges.items():
        u, w = info['dir']
        for c in (0, 1):
            uc, wc = v_copy(u, c), v_copy(w, c)
            out_edges.append((uc, wc))
            if info['is_rib']:
                # Attaching region edge! Wait: what if its face-degree in Y ends up being 2
                # (e.g. at a boundary edge of X_{(i)} where only 1 face of X' plus 1 cylinder face meet)?
                # Let's check BDL line 5034 (fig:quantum_doubling_corner) and line 4221:
                # In Y, ANY edge whose final face-degree == 2 gets 'none' ({00, 11})!
                # And if its final face-degree != 2 (e.g. 3 inside a face: 2 faces in X' + 1 face in X_cyl!),
                # attaching region edges get `new_sheaf_color` ('blue' for R_X, 'red' for R_Z)!
                out_colors.append(new_sheaf_color)
            else:
                out_colors.append(info['color'])

    # Add faces of Copy 0 and Copy 1
    for (v0, v1, v2, v3) in xp_faces:
        for c in (0, 1):
            out_faces.append((v_copy(v0, c), v_copy(v1, c), v_copy(v2, c), v_copy(v3, c)))

    # Add vertical cylinder edges v_0 -> v_1 for each rib vertex v in rib_verts
    # Wait: what color does a vertical cylinder edge v_0 -> v_1 get?!
    # READ BDL LINES 5056-5062 VERY CAREFULLY!!
    # "Edges of the second type [edges in X_{cyl}, including attaching regions and interior edges of X_{cyl}]
    #  are, in most cases, colored blue in the iteration R_X and red in the iteration R_Z.
    #  The ONLY exception occurs in the doubling of a T-junction, where the edge along O-P-Q
    #  inherits the color of the degree-3 edge of the original T-junction."
    # WAIT A SECOND!! CHECK THAT EXCEPTION ON LINE 5059!!
    # What IS "the edge along O-P-Q" in the doubling of a T-junction (or any original edge e of X_{(i)} with face-degree != 2)?!
    # When ribs in X' meet an original edge e of X_{(i)} at a subdivided interior vertex v of e:
    #   - If e had face-degree 3 (T-junction, colored c_orig in {'blue', 'red'}), then 3 ribs (one on each of the 3 faces incident to e)
    #     meet at v! So in X_{cyl}, THREE cylinder faces meet along the vertical cylinder edge (v_0 -> v_1)!!
    #     AND BDL line 5060 says: THAT vertical cylinder edge (v_0 -> v_1) MUST INHERIT THE COLOR OF e (`c_orig`),
    #     NOT `new_sheaf_color`!!
    #   - What if e was a BOUNDARY edge of X_{(i)} (face-degree 1, colored c_orig in {'blue', 'red'})?!
    #     CHECK THIS!! Suppose e is a boundary edge of X_{(0)} (face-degree 1, say 'blue' horizontal boundary or 'red' vertical boundary)!
    #     A perpendicular rib inside the square hits e at vertex v!
    #     Then in X_{cyl}, ONLY ONE cylinder face meets the vertical cylinder edge (v_0 -> v_1) — so (v_0 -> v_1) is a BOUNDARY edge of Y (face-degree 1)!!
    #     Which boundary of Y does (v_0 -> v_1) belong to?! It connects Copy 0's boundary e to Copy 1's boundary e — it IS part of the thickened boundary wall of e!!
    #     WAIT: look at BDL Section 3.2 (`replacement-Cx.pdf` / `intro-step-24` / Remark 2 line 1470):
    #     Along an original boundary edge e of color `c_orig` (blue = e-condensing, red = m-condensing),
    #     the thickened boundary wall is STILL `c_orig`-condensing!!
    #     Let's check! Look at `fig:quantum_doubling_plane` (line 5014, boundary = subcomplex of T-junction with 1 face instead of 3!):
    #     In `fig:quantum_doubling_plane`, O-P-Q is the boundary edge (degree 1) of a single plane, which BDL explicitly treats as a subcomplex of the T-junction — so (v_0 -> v_1) INHERITS `c_orig` whenever v lies on an original edge e with face-degree != 2!!
    #     Whereas when v is strictly in the INTERIOR of a face of X_{(i)} (a glued cross where horizontal & vertical ribs cross, so 4 cylinder faces meet at v_0 -> v_1!), v does NOT lie on any original edge e, so (v_0 -> v_1) gets `new_sheaf_color`!!
    #     Wait: what if v lies on an original edge e of X_{(i)} that had face-degree == 2?!
    #     Then 2 ribs meet at v, so 2 cylinder faces meet at (v_0 -> v_1) -> face-degree 2 -> 'none' ({00, 11}) automatically!!

    # Let's map each vertex of X' to any original edge e_idx it belongs to (interior subdivided vertices of e belong to a UNIQUE original edge e_idx!)
    xp_vert_orig_edge = {}
    for e_idx, pts in edge_pts.items():
        for v in pts[1:-1]:
            xp_vert_orig_edge[v] = e_idx

    for v in sorted(rib_verts):
        v0, v1 = v_copy(v, 0), v_copy(v, 1)
        out_edges.append((v0, v1))
        if v in xp_vert_orig_edge:
            e_idx = xp_vert_orig_edge[v]
            fdeg = len(complex_in.edge_faces[e_idx])
            if fdeg != 2:
                out_colors.append(complex_in.edge_colors[e_idx])
            else:
                out_colors.append('none')
        else:
            # Interior of a face (glued cross of ribs -> face-degree 4 in X_cyl)
            out_colors.append(new_sheaf_color)

    # Add cylinder faces above each rib edge (u -> w) in X'
    for key, info in xp_edges.items():
        if info['is_rib']:
            u, w = info['dir']
            # Bottom: u0 -> w0, Top: u1 -> w1, Left: u0 -> u1, Right: w0 -> w1
            out_faces.append((v_copy(u, 0), v_copy(w, 0), v_copy(w, 1), v_copy(u, 1)))

    # Normalize: any edge in out_edges that ends up with face-degree == 2 has color 'none'
    prelim = CellComplex2D(total_verts, out_edges, out_colors, out_faces)
    for e_idx in range(len(prelim.edges)):
        if len(prelim.edge_faces[e_idx]) == 2:
            prelim.edge_colors[e_idx] = 'none'
        else:
            assert prelim.edge_colors[e_idx] in ('blue', 'red'), (
                f"Edge {e_idx} has face-degree {len(prelim.edge_faces[e_idx])} != 2 but color='none'!"
            )
    return prelim


def build_css_from_sheaf(comp: CellComplex2D):
    """
    Construct exact CSS parity check matrices H_X, H_Z from C^0(X, F) -> C^1(X, F) -> C^2(X, F).
    Returns:
      num_qubits, H_X (list of int bitmasks or list of qubit lists), H_Z (list of qubit lists)
    """
    # 1. C^2(X, F): 1 basis element per face f in comp.faces
    num_faces = len(comp.faces)

    # 2. C^1(X, F): for each edge e, local subspace F_e subset F_2^{edge_faces[e]}
    #    Represent each qubit as a pair (e_idx, vec_on_incident_faces)
    edge_qubit_start = []
    qubits = []  # list of dict {face_idx: 1}
    edge_local_subspace = []  # for each edge, list of basis dicts {f_idx: 1}

    for e_idx, inc_faces in enumerate(comp.edge_faces):
        d = len(inc_faces)
        col = comp.edge_colors[e_idx]
        edge_qubit_start.append(len(qubits))
        basis = []
        if d == 0:
            pass
        elif d == 2 or col == 'blue':
            # Repetition code {0^d, 1^d}: 1 basis vector = all 1s on inc_faces
            vec = {f: 1 for f in inc_faces}
            basis.append(vec)
        elif col == 'red':
            # Parity-check code {even weight on inc_faces}: dim = d - 1
            # Use adjacent pairs (inc_faces[j], inc_faces[j+1]) for j=0..d-2
            for j in range(d - 1):
                vec = {inc_faces[j]: 1, inc_faces[j + 1]: 1}
                basis.append(vec)
        else:
            raise ValueError(f"Unexpected edge {e_idx}: d={d}, col={col}")
        edge_local_subspace.append(basis)
        for vec in basis:
            qubits.append((e_idx, vec))

    num_qubits = len(qubits)

    # H_Z: 1 row per face tau, support = qubits whose vec has tau: 1
    H_Z = [[] for _ in range(num_faces)]
    for q_idx, (e_idx, vec) in enumerate(qubits):
        for f_idx in vec:
            H_Z[f_idx].append(q_idx)

    # 3. C^0(X, F): for each vertex v, F_v subset F_2^{vert_faces[v]}
    #    defined by strong sheaf condition (Eq. 6 of BDL):
    #    alpha in F_2^{vert_faces[v]} s.t. for every e in vert_edges[v],
    #      alpha|_{edge_faces[e]} in F_e!
    #    Let's translate each condition precisely:
    #      - If e is 'blue' or d_e == 2: all faces in edge_faces[e] must have EQUAL value in alpha!
    #      - If e is 'red' (and d_e != 2): sum of alpha over faces in edge_faces[e] == 0 (mod 2)!
    #        (In particular, if d_e == 1 and 'red', that single face must have alpha_f == 0!)
    #    AND once we have a basis of F_v, how does alpha in F_v map via delta^0 to C^1(X, F) = bigoplus_e F_e?
    #    For each e in vert_edges[v], alpha|_{edge_faces[e]} lies in F_e, so we express it uniquely
    #    in our chosen basis `edge_local_subspace[e]` of F_e!!
    H_X = []
    for v in range(comp.num_vertices):
        v_faces = sorted(set(comp.vert_faces[v]))
        if not v_faces:
            continue
        m = len(v_faces)
        f_to_local = {f: i for i, f in enumerate(v_faces)}

        # Build linear constraint matrix over F_2 on m variables
        constraints = []
        for e_idx in comp.vert_edges[v]:
            inc = comp.edge_faces[e_idx]
            d = len(inc)
            col = comp.edge_colors[e_idx]
            if d == 2 or col == 'blue':
                # All faces in inc equal: inc[0] + inc[j] = 0 for j=1..d-1
                for j in range(1, d):
                    row = 0
                    row ^= (1 << f_to_local[inc[0]])
                    row ^= (1 << f_to_local[inc[j]])
                    constraints.append(row)
            elif col == 'red':
                # Sum of all faces in inc == 0
                row = 0
                for f in inc:
                    row ^= (1 << f_to_local[f])
                constraints.append(row)

        # Find nullspace of `constraints` over F_2^m
        # Since m <= ~16 (bounded vertex degree!), we can do exact Gaussian elimination
        pivots = {}
        for r in constraints:
            x = r
            while x:
                lsb = (x & -x).bit_length() - 1
                if lsb in pivots:
                    x ^= pivots[lsb]
                else:
                    pivots[lsb] = x
                    break
        free_vars = [i for i in range(m) if i not in pivots]
        if not free_vars:
            continue

        # Construct canonical nullspace basis vectors (back-substituted so pivots are reduced!)
        # Reduce pivot rows upward too (RREF) for sparsest basis!
        for p in sorted(pivots.keys(), reverse=True):
            for p2 in pivots:
                if p2 != p and ((pivots[p2] >> p) & 1):
                    pivots[p2] ^= pivots[p]

        for fv in free_vars:
            alpha_mask = (1 << fv)
            for p, prow in pivots.items():
                if (prow >> fv) & 1:
                    alpha_mask |= (1 << p)

            # Map alpha_mask to qubits on vert_edges[v]
            x_supp = []
            for e_idx in comp.vert_edges[v]:
                inc = comp.edge_faces[e_idx]
                d = len(inc)
                col = comp.edge_colors[e_idx]
                q0 = edge_qubit_start[e_idx]
                if d == 0:
                    continue
                elif d == 2 or col == 'blue':
                    # Value on inc[0]
                    if (alpha_mask >> f_to_local[inc[0]]) & 1:
                        x_supp.append(q0)
                elif col == 'red':
                    # Express even-weight vector (b_0, ..., b_{d-1}) in basis (1,1,0..), (0,1,1,0..), ...
                    # Coeff c_j = sum_{s=0}^{j} b_s (mod 2)!
                    # Check: c_0*(1,1,0..) + c_1*(0,1,1..) -> pos 0 is c_0=b_0, pos 1 is c_0+c_1=b_1, ...,
                    # pos d-1 is c_{d-2} = b_0+...+b_{d-2} = b_{d-1} (since sum b_s = 0)! EXACT!
                    pref = 0
                    for j in range(d - 1):
                        pref ^= ((alpha_mask >> f_to_local[inc[j]]) & 1)
                        if pref:
                            x_supp.append(q0 + j)
            if x_supp:
                H_X.append(sorted(x_supp))

    return num_qubits, H_X, H_Z


def gf2_rank(num_cols, rows):
    """Exact GF(2) rank using Python arbitrary-precision integers."""
    pivots = {}
    for supp in rows:
        x = 0
        for c in supp:
            x ^= (1 << c)
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in pivots:
                x ^= pivots[lsb]
            else:
                pivots[lsb] = x
                break
    return len(pivots), pivots


def check_commutativity(H_X, H_Z):
    """Verify H_X * H_Z^T == 0 mod 2."""
    z_sets = [set(r) for r in H_Z]
    # Use inverted index for speed on large codes
    q_to_z = defaultdict(list)
    for zi, r in enumerate(H_Z):
        for q in r:
            q_to_z[q].append(zi)
    for xi, rx in enumerate(H_X):
        counts = defaultdict(int)
        for q in rx:
            for zi in q_to_z[q]:
                counts[zi] ^= 1
        for zi, parity in counts.items():
            if parity != 0:
                return False, (xi, zi)
    return True, None


def exact_min_distance_small(num_q, H_check, H_stab):
    """
    Exact minimum weight of c in ker(H_check) \ rowspan(H_stab) for small codes (num_q <= 35),
    or upper bound / BFS for moderate codes.
    """
    _, check_pivots = gf2_rank(num_q, H_check)
    # RREF of check_pivots
    for p in sorted(check_pivots.keys(), reverse=True):
        for p2 in check_pivots:
            if p2 != p and ((check_pivots[p2] >> p) & 1):
                check_pivots[p2] ^= check_pivots[p]
    free_vars = [i for i in range(num_q) if i not in check_pivots]
    ker_basis = []
    for fv in free_vars:
        vec = (1 << fv)
        for p, prow in check_pivots.items():
            if (prow >> fv) & 1:
                vec |= (1 << p)
        ker_basis.append(vec)

    rank_stab, stab_pivots = gf2_rank(num_q, H_stab)

    def in_stab(vec):
        x = vec
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in stab_pivots:
                x ^= stab_pivots[lsb]
            else:
                return False
        return True

    # Find quotient basis of ker(H_check) / rowspan(H_stab)
    quot_pivots = dict(stab_pivots)
    logicals = []
    for v in ker_basis:
        x = v
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in quot_pivots:
                x ^= quot_pivots[lsb]
            else:
                quot_pivots[lsb] = x
                logicals.append(v)
                break
    if len(logicals) != 1:
        return f"k={len(logicals)}"
    L0 = logicals[0]

    # Coset L0 + rowspan(H_stab): if dim(stab) <= 18, enumerate all 2^dim(stab) exactly!
    stab_vecs = list(stab_pivots.values())
    if len(stab_vecs) <= 18:
        best = bin(L0).count("1")
        # Gray code enumeration
        cur = L0
        for i in range(1, 1 << len(stab_vecs)):
            bit = (i & -i).bit_length() - 1
            cur ^= stab_vecs[bit]
            w = bin(cur).count("1")
            if w < best:
                best = w
        return best
    return "?"


if __name__ == "__main__":
    for label, L, ribs in [
        ("1-rib minimal (L=2, ribs=[1])", 2, [1]),
        ("1-rib centered (L=3, ribs=[1])", 3, [1]),
        ("2-rib minimal (L=3, ribs=[1,2])", 3, [1, 2]),
        ("2-rib symmetric (L=4, ribs=[1,2])", 4, [1, 2]),
        ("BDL explicit scale (L=8, ribs=[4,5])", 8, [4, 5]),
    ]:
        print(f"\n================ {label} ================")
        comp = make_seed_single_square()
        max_steps = 4 if L <= 3 else (3 if L == 4 else 2)
        for step in range(max_steps + 1):
            if step > 0:
                itype = 'X' if (step % 2 == 1) else 'Z'
                comp = apply_iteration(comp, L, ribs, itype)
            nq, HX, HZ = build_css_from_sheaf(comp)
            ok, bad = check_commutativity(HX, HZ)
            assert ok, f"Commutativity failed at step {step}: {bad}"
            max_wx = max((len(r) for r in HX), default=0)
            max_wz = max((len(r) for r in HZ), default=0)
            q_deg_x = [0] * nq
            q_deg_z = [0] * nq
            for r in HX:
                for q in r:
                    q_deg_x[q] += 1
            for r in HZ:
                for q in r:
                    q_deg_z[q] += 1
            max_qx = max(q_deg_x, default=0)
            max_qz = max(q_deg_z, default=0)

            if nq <= 4000:
                rx, _ = gf2_rank(nq, HX)
                rz, _ = gf2_rank(nq, HZ)
                k_log = nq - rx - rz
                dx = exact_min_distance_small(nq, HZ, HX) if rx <= 18 else "?"
                dz = exact_min_distance_small(nq, HX, HZ) if rz <= 18 else "?"
                print(
                    f"  step {step} ({'seed' if step==0 else ('R_X' if step%2==1 else 'R_Z')}): "
                    f"faces={len(comp.faces):6d}, [[n={nq:6d}, k={k_log}, d=({dx},{dz})]], "
                    f"|HX|={len(HX):5d} (rk {rx}), |HZ|={len(HZ):5d} (rk {rz}), "
                    f"max_w=({max_wx},{max_wz}), max_qdeg=({max_qx},{max_qz})"
                )
            else:
                print(
                    f"  step {step} ({'R_X' if step%2==1 else 'R_Z'}): "
                    f"faces={len(comp.faces):6d}, n={nq:6d}, |HX|={len(HX):6d}, |HZ|={len(HZ):6d}, "
                    f"max_w=({max_wx},{max_wz}), max_qdeg=({max_qx},{max_qz})"
                )
