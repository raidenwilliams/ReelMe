import requests
import streamlit as st
import csv
from streamlit_lottie import st_lottie
from streamlit_agraph import agraph, Node, Edge, Config
import heapq
import time


# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="ReelMe", page_icon="🍿", layout="centered")


def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_bb9bkg1h.json")
lottie_coding1 = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_NEId4v1Sv3.json")


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")


### BACKEND
def find_movies(edge_data):
    uniques = []
    for edge in edge_data:
        if not edge[0] in uniques:
            uniques.append(edge[0])
        if not edge[1] in uniques:
            uniques.append(edge[1])
    return uniques


def read_edge_file(filename):
    edges = []
    with open(filename, encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            from_movie = row[0]
            to_movie = row[1]
            weight = row[2]
            edges.append([from_movie, to_movie, weight])
    return edges


def read_movie_file(filename):
    movies = []
    with open(filename, encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            movie = row[0]
            movies.append(movie)
    return movies


movie_displays = read_movie_file("data/restricted_uniques.csv")


# ------------ USE FRONT-END  -------------
# ADJACENCY MATRIX DATA STRUCTURE AND SEARCH ALGORITHM
class AdjacencyMatrix:
    def __init__(self, edges, movies):
        self.graph: dict[str, dict[str, float]]
        self.graph = {}
        for movie in movies:
            self.graph[movie] = {}
        for row in self.graph:
            for movie in movies:
                self.graph[row][movie] = 0.0
        for edge in edges:
            self.add_connection(edge[0], edge[1], edge[2])

    def add_connection(self, from_vertex, to_vertex, weight):
        if from_vertex not in self.graph:
            self.graph[from_vertex] = {}
            self.graph[from_vertex][from_vertex] = 0.0
        if to_vertex not in self.graph:
            self.graph[to_vertex] = {}
            self.graph[to_vertex][to_vertex] = 0.0
        self.graph[from_vertex][to_vertex] = weight
        self.graph[to_vertex][from_vertex] = weight

    def print_connections(self):
        for v in self.graph:
            print(v)
            print(self.graph[v])

    def bidirectional_search(self, targets):
        hits = {}
        visits = {}
        distances = {}
        previous = {}
        heap = {}

        num_to_hit = len(targets)

        middle_point = "null"

        for root in targets:
            previous[root] = {}
            distances[root] = {}
            distances[root][root] = -1
            hits[root] = 0
            heap[root] = [(-1, root)]
            visits[root] = []

        while True:
            for root in targets:
                #print("Dijking", root)
                (minimum_weight, choice) = heapq.heappop(heap[root])
                empty = False
                if choice in visits[root] or (choice in targets and choice != root):
                    if len(heap[root]) == 0:
                        continue
                    while choice in visits[root] or choice in targets:
                        (minimum_weight, choice) = heapq.heappop(heap[root])
                        if len(heap[root]) == 0:
                            empty = True
                            break
                if empty:
                    continue
                visits[root].append(choice)
                if choice not in hits:
                    hits[choice] = 1
                else:
                    hits[choice] += 1
                    if hits[choice] == num_to_hit:
                        middle_point = choice
                #print("Current vertex:",choice)
                for adjacent in self.graph[choice]:
                    if self.graph[choice][adjacent] == float(0.0):
                        continue
                    weight = self.graph[choice][adjacent]
                    distance = float(minimum_weight) + float(weight)
                    if adjacent not in distances[root] or distance < distances[root][adjacent]:
                        distances[root][adjacent] = distance
                        heapq.heappush(heap[root], (distance, adjacent))
                        previous[root][adjacent] = choice
            if middle_point != "null":
                break

        print("Found it:", middle_point)
        paths = {}
        for root in targets:
            # print("Root:",root)
            paths[root] = []
            next_in_path = middle_point
            while next_in_path != root:
                # ("Appending", previous[root][next_in_path], next_in_path)
                weight = distances[root][next_in_path] - distances[root][previous[root][next_in_path]]
                paths[root].append((previous[root][next_in_path], next_in_path, weight))
                next_in_path = previous[root][next_in_path]

        edges = []
        for path in paths:
            for edge in paths[path]:
                if edge not in edges:
                    edges.append(edge)

        for edge in edges:
            print(edge[0], "->", edge[1], ":", edge[2])

        return [edges, middle_point]


# ADJACENCY LIST DATA STRUCTURE AND SEARCH ALGORITHM
class AdjacencyList:

    def __init__(self, edges):
        self.graph: dict[str, list]
        self.graph = {}
        for edge in edges:
            self.add_connection(str(edge[0]), str(edge[1]), float(edge[2]))

    def add_connection(self, from_vertex, to_vertex, weight):
        if from_vertex not in self.graph:
            self.graph[from_vertex] = []
        if to_vertex not in self.graph:
            self.graph[to_vertex] = []
        self.graph[from_vertex].append((str(to_vertex), float(weight)))
        self.graph[to_vertex].append((str(from_vertex), float(weight)))

    def print_connections(self):
        for v in self.graph:
            print(v)
            print(len(self.graph[v]))

    def bidirectional_search(self, targets):
        hits = {}
        visits = {}
        distances = {}
        previous = {}
        heap = {}

        middle_point = "null"

        for root in targets:
            previous[root] = {}
            distances[root] = {}
            distances[root][root] = 0
            hits[root] = 0
            heap[root] = [(0, root)]
            visits[root] = []

        while True:
            for root in targets:
                #print("Dijking", root)
                (minimum_weight, choice) = heapq.heappop(heap[root])
                if choice in visits[root] or (choice in targets and choice != root):
                    if len(heap[root]) == 0:
                        continue
                    while choice in visits[root] or choice in targets:
                        (minimum_weight, choice) = heapq.heappop(heap[root])
                        if len(heap[root]) == 0:
                            empty = True
                            break
                visits[root].append(choice)
                if choice not in hits:
                    hits[choice] = 1
                else:
                    hits[choice] += 1
                    if hits[choice] == len(targets):
                        middle_point = choice
                #print("Current vertex:",choice)
                #print("graph:", self.graph[choice])
                for i in range(0, len(self.graph[choice])):
                    adjacent = self.graph[choice][i][0]
                    weight = self.graph[choice][i][1]
                    distance = minimum_weight + weight
                    if adjacent not in distances[root] or distance < distances[root][adjacent]:
                        distances[root][adjacent] = distance
                        heapq.heappush(heap[root], (distance, adjacent))
                        previous[root][adjacent] = choice
            if middle_point != "null":
                break

        print("Found it:", middle_point)
        paths = {}
        for root in targets:
            #print("Root:",root)
            paths[root] = []
            next_in_path = middle_point
            while next_in_path != root:
                #("Appending", previous[root][next_in_path], next_in_path)
                weight = distances[root][next_in_path] - distances[root][previous[root][next_in_path]]
                paths[root].append((previous[root][next_in_path], next_in_path, weight))
                next_in_path = previous[root][next_in_path]

        edges = []
        for path in paths:
            for edge in paths[path]:
                if edge not in edges:
                    edges.append(edge)

        for edge in edges:
            print(edge[0],"->",edge[1],":",edge[2])

        return [edges, middle_point]


def fix_movie_name(name):
    name = name.capitalize()
    fixed = ''
    for i in range(0, len(name)):
        if name[i] == "-":
            fixed = fixed + " "
        elif name[i-1] == "-":
            fixed = fixed + name[i].upper()
        else:
            fixed = fixed + name[i]

    return fixed


# Method to create agraph graph
def create_graph(movie_data_, edge_data_, queries, recommendation):
    nodes = []
    edges = []
    for movie in movie_data_:
        if movie in queries:
            nodes.append(Node(id=str(movie),
                              label=str(movie),
                              size=10,
                              shape='square')
                         )
        elif movie == recommendation:
            nodes.append(Node(id=str(movie),
                              label=str(movie),
                              size=15)
                         )
        else:
            nodes.append(Node(id=str(movie),
                              label=str(movie),
                              size=5)
                         )
    for edge in edge_data_:
        print("Adding edge", edge[0], edge[1])
        edges.append(Edge(source=str(edge[0]),
                          label=str(round(float(edge[2]), 1)),
                          target=str(edge[1]),
                          nodeHighlightBehavior=True
                          )
                     )
    config = Config(width=600,
                    height=350,
                    directed=False,
                    physics=True,
                    hierarchical=False,
                    # **kwargs
                    )

    return [nodes, edges, config]


def display_graph(nodes_, edges_, config_):
    agraph(nodes=nodes_, edges=edges_, config=config_)

def find_image(filename, recommendation):
    with open(filename, encoding = "utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            movie = row[0]
            if movie == recommendation:
                image = row[1]
                return image
    return None

def generate_graph(query_, implementation):
    print("Finding recommendation")
    if st.session_state.density_set == '100,000 Edges':
        edge_data = read_edge_file("data/sparse.csv")
    else:
        edge_data = read_edge_file("data/dense.csv")
    if implementation == "Adjacency List" or implementation == 'Both':
        st.session_state.list_start = time.time()
        print("Building Adjacency List")

        st.session_state.list_construction_start = time.time()
        myList = AdjacencyList(edge_data)
        st.session_state.list_construction_end = time.time()

        st.session_state.list_search_start = time.time()
        (recommended_edges, recommendation) = myList.bidirectional_search(query)
        st.session_state.list_search_end = time.time()

    if implementation == 'Adjacency Matrix' or implementation == 'Both':
        print("Building Adjacency Matrix")
        st.session_state.mat_start = time.time()
        st.session_state.mat_construction_start = time.time()
        myMatrix = AdjacencyMatrix(edge_data, find_movies(edge_data))
        st.session_state.mat_construction_end = time.time()

        st.session_state.mat_search_start = time.time()
        (recommended_edges, recommendation) = myMatrix.bidirectional_search(query)
        st.session_state.mat_search_end = time.time()

    print("Recommendation:",recommendation)
    st.title("Your movie recommendation is:")
    st.title("👉" + recommendation)
    st.session_state.image_link = find_image("data/restricted_uniques.csv", recommendation)
    movie_data = find_movies(recommended_edges)
    for movie in movie_data:
        print("Movie:",movie)
    for edge in recommended_edges:
        print("Edge:",edge)
    print("Creating graph")
    graph_data = create_graph(movie_data, recommended_edges, query_, recommendation)
    st.session_state.list_end = time.time()
    st.session_state.mat_end = time.time()
    print("Displaying graph")
    return graph_data, recommendation


pressed = False

st.title("🍿ReelMe")
st.subheader("About Us")
st.markdown(
    "**Welcome to ReelMe! Using our selection of 3,000 unique movies, simply enter you and your friends' favorite movies and we'll do the rest. No matter how diverse your movie tastes, we can find the shortest path to a memorable movie night.✌️**")
left_column, right_column = st.columns(2)
with left_column:
    st.text("")
    st.subheader("How to use ReelMe 👇")
    st.markdown(
        "1. Enter your favorite movies.\n2. Click **Create Graph** to get your recommendation.\n3. Enjoy your movie night!")
with right_column:
    # lotte file images
    st_lottie(lottie_coding, speed=1, height=200, key="initial")

if "i_am_ready" not in st.session_state:
    st.session_state.i_am_ready = False

click = st.button("I am ready!")

if click:
    st.session_state.i_am_ready = True

if "pressed" not in st.session_state:
    st.session_state.pressed = False

if "graph_to_display" not in st.session_state:
    st.session_state.graph_to_display = graph_to_display = create_graph([], [], [], None)

if not st.session_state.pressed:
    graph_to_display = create_graph([], [], [], None)

if "list_start" not in st.session_state:
    st.session_state.list_start = 0
if "list_end" not in st.session_state:
    st.session_state.list_end = 0
if "list_construction_start" not in st.session_state:
    st.session_state.list_construction_start = 0
if "list_construction_end" not in st.session_state:
    st.session_state.list_construction_end = 0
if "list_search_start" not in st.session_state:
    st.session_state.list_search_start = 0
if "list_search_end" not in st.session_state:
    st.session_state.list_search_end = 0

if "mat_start" not in st.session_state:
    st.session_state.mat_start = 0
if "mat_end" not in st.session_state:
    st.session_state.mat_end = 0
if "mat_construction_start" not in st.session_state:
    st.session_state.mat_construction_start = 0
if "mat_construction_end" not in st.session_state:
    st.session_state.mat_construction_end = 0
if "mat_search_start" not in st.session_state:
    st.session_state.mat_search_start = 0
if "mat_search_end" not in st.session_state:
    st.session_state.mat_search_end = 0

if "graph_set" not in st.session_state:
    st.session_state.graph_set = ''

if "density_set" not in st.session_state:
    st.session_state.density_set = ''

if "query_length" not in st.session_state:
    st.session_state.query_length = 0

if "image_link" not in st.session_state:
    st.session_state.image_link = ''

# input page
if st.session_state.i_am_ready:
    st.divider()
    col = st.columns(2)
    with col[0]:
        st.text("")
        st.title("Get Started")
        innercol = st.columns(2)
        with (innercol[0]):
            graph_type = st.radio(
                "Select the graph implementation to use:",
                ('Adjacency List', 'Adjacency Matrix', 'Both'))
        with (innercol[1]):
            density = st.radio(
                "Select the graph density to use:",
                ('100,000 Edges', '1,000,000 Edges'))
    with col[1]:
        st_lottie(lottie_coding1, speed=.5, height=200, key="initial1")

    options = st.multiselect(
        'Your favorite movies...',
        movie_displays
    )

    # RUNTIME
    query = options


    if st.button('Create graph'):
        st.session_state.query_length = len(query)
        st.session_state.graph_set = graph_type
        st.session_state.density_set = density
        st.session_state.pressed = True
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            latest_iteration.text(f'Calculating... {i + 1}%')
            bar.progress(i + 1)
            time.sleep(0.01)
        print("Generating graph")
        st.session_state.graph_to_display, recommendation = generate_graph(query, st.session_state.graph_set)
        st.session_state.listend = time.time()

    options = []

image, graph = st.columns([1, 2])
with image:
    if st.session_state.image_link != '':
        st.image(st.session_state.image_link)
with graph:
    agraph(nodes=st.session_state.graph_to_display[0], edges=st.session_state.graph_to_display[1],
          config=st.session_state.graph_to_display[2])

if st.session_state.pressed:
    st.write("Execution Time (in seconds)")
    list_construction = float(st.session_state.list_construction_end - st.session_state.list_construction_start)
    list_search = float(st.session_state.list_search_end - st.session_state.list_search_start)
    list_total = float(st.session_state.list_end - st.session_state.list_start)
    list_other = list_total - (list_search + list_construction)

    mat_construction = float(st.session_state.mat_construction_end - st.session_state.mat_construction_start)
    mat_search = float(st.session_state.mat_search_end - st.session_state.mat_search_start)
    mat_total = float(st.session_state.mat_end - st.session_state.mat_start)
    mat_other = mat_total - (mat_search + mat_construction)


    if st.session_state.graph_set == 'Adjacency List':
        st.metric(label="Adjacency List Construction",
                  value=list_construction)
        st.metric(label="Adjacency List " + str(st.session_state.query_length) + "-Directional Search", value=list_search)
        st.metric(label="Other (visualization, etc)", value=list_other)
        st.metric(label="Total", value=list_total)
    elif st.session_state.graph_set == 'Adjacency Matrix':
        st.metric(label="Adjacency Matrix Construction",
                  value=mat_construction)
        st.metric(label="Adjacency Matrix " + str(st.session_state.query_length) + "-Directional Search",
                  value=mat_search)
        st.metric(label="Other (visualization, etc)", value=mat_other)
        st.metric(label="Total", value=mat_total)
    else:
        timecol = st.columns(2)

        list_total = (list_total - mat_total)
        list_other = list_other - (st.session_state.mat_search_end - st.session_state.mat_construction_start)
        mat_other = list_other

        with timecol[0]:
            st.metric(label="Adjacency List Construction",
                      value=list_construction)
            st.metric(label="Adjacency List " + str(st.session_state.query_length) + "-Directional Search",
                      value=list_search)
            st.metric(label="Other (visualization, etc)", value=list_other)
            st.metric(label="Total", value=list_total)
        with timecol[1]:
            st.metric(label="Adjacency Matrix Construction",
                      value=mat_construction)
            st.metric(label="Adjacency Matrix " + str(st.session_state.query_length) + "-Directional Search",
                      value=mat_search)
            st.metric(label="Other (visualization, etc)", value=mat_other)
            st.metric(label="Total", value=mat_total)


st.markdown(
    "This is a movie recommendation tool created by **Shawn Rhoads, Raiden Williams, and Brody Foster** using an N-Directional search algorithm on a Letterbox movie review dataset by Sam Learner.")

