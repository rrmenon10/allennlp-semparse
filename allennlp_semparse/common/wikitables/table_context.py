import re
import csv
from typing import Union, Dict, List, Tuple, Set
from collections import defaultdict

from unidecode import unidecode
from allennlp.data.tokenizers import Token

from allennlp_semparse.common import Date, NUMBER_CHARACTERS, NUMBER_WORDS, ORDER_OF_MAGNITUDE_WORDS
from allennlp_semparse.common.knowledge_graph import KnowledgeGraph

# == stop words that will be omitted by ContextGenerator
STOP_WORDS = {
    "",
    "",
    "all",
    "being",
    "-",
    "over",
    "through",
    "yourselves",
    "its",
    "before",
    "hadn",
    "with",
    "had",
    ",",
    "should",
    "to",
    "only",
    "under",
    "ours",
    "has",
    "ought",
    "do",
    "them",
    "his",
    "than",
    "very",
    "cannot",
    "they",
    "not",
    "during",
    "yourself",
    "him",
    "nor",
    "did",
    "didn",
    "'ve",
    "this",
    "she",
    "each",
    "where",
    "because",
    "doing",
    "some",
    "we",
    "are",
    "further",
    "ourselves",
    "out",
    "what",
    "for",
    "weren",
    "does",
    "above",
    "between",
    "mustn",
    "?",
    "be",
    "hasn",
    "who",
    "were",
    "here",
    "shouldn",
    "let",
    "hers",
    "by",
    "both",
    "about",
    "couldn",
    "of",
    "could",
    "against",
    "isn",
    "or",
    "own",
    "into",
    "while",
    "whom",
    "down",
    "wasn",
    "your",
    "from",
    "her",
    "their",
    "aren",
    "there",
    "been",
    ".",
    "few",
    "too",
    "wouldn",
    "themselves",
    ":",
    "was",
    "until",
    "more",
    "himself",
    "on",
    "but",
    "don",
    "herself",
    "haven",
    "those",
    "he",
    "me",
    "myself",
    "these",
    "up",
    ";",
    "below",
    "'re",
    "can",
    "theirs",
    "my",
    "and",
    "would",
    "then",
    "is",
    "am",
    "it",
    "doesn",
    "an",
    "as",
    "itself",
    "at",
    "have",
    "in",
    "any",
    "if",
    "!",
    "again",
    "'ll",
    "no",
    "that",
    "when",
    "same",
    "how",
    "other",
    "which",
    "you",
    "many",
    "shan",
    "'t",
    "'s",
    "our",
    "after",
    "most",
    "'d",
    "such",
    "'m",
    "why",
    "a",
    "off",
    "i",
    "yours",
    "so",
    "the",
    "having",
    "once",
}

CellValueType = Union[str, float, Date]


class TableContext:
    """
    Representation of table context similar to the one used by Memory Augmented Policy Optimization (MAPO, Liang et
    al., 2018). Most of the functionality is a reimplementation of
    https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/table/wtq/preprocess.py
    for extracting entities from a question given a table and type its columns with <string> | <date> | <number>
    """

    def __init__(
        self,
        table_data: List[Dict[str, CellValueType]],
        column_name_type_mapping: Dict[str, Set[str]],
        column_names: Set[str],
    ) -> None:
        self.table_data = table_data
        self.column_types: Set[str] = set()
        self.column_names = column_names
        for types in column_name_type_mapping.values():
            self.column_types.update(types)
        # Mapping from strings to the columns they are under.
        string_column_mapping: Dict[str, List[str]] = defaultdict(list)
        for table_row in table_data:
            for column_name, cell_value in table_row.items():
                if "string_column:" in column_name and cell_value is not None:
                    string_column_mapping[str(cell_value)].append(column_name)
        # We want the object to raise KeyError when checking if a specific string is a cell in the
        # table.
        self._string_column_mapping = dict(string_column_mapping)
        self._table_knowledge_graph: KnowledgeGraph = None

    def __eq__(self, other):
        if not isinstance(other, TableQuestionContext):
            return False
        return self.table_data == other.table_data

    def get_table_knowledge_graph(self) -> KnowledgeGraph:
        if self._table_knowledge_graph is None:
            entities: Set[str] = set()
            neighbors: Dict[str, List[str]] = defaultdict(list)
            entity_text: Dict[str, str] = {}
            # Add all column names to entities. We'll define their neighbors to be empty lists for
            # now, and later add number and string entities as needed.
            number_columns = []
            date_columns = []
            for typed_column_name in self.column_names:
                if "number_column:" in typed_column_name or "num2_column" in typed_column_name:
                    number_columns.append(typed_column_name)

                if "date_column:" in typed_column_name:
                    date_columns.append(typed_column_name)

                # Add column names to entities, with no neighbors yet.
                entities.add(typed_column_name)
                neighbors[typed_column_name] = []
                entity_text[typed_column_name] = typed_column_name.split(":", 1)[-1].replace(
                    "_", " "
                )

            # string_entities, numbers = self.get_entities_from_question()
            # for entity, column_names in string_entities:
            #     entities.add(entity)
            #     for column_name in column_names:
            #         neighbors[entity].append(column_name)
            #         neighbors[column_name].append(entity)
            #     entity_text[entity] = entity.replace("string:", "").replace("_", " ")
            # # For all numbers (except -1), we add all number and date columns as their neighbors.
            # for number, _ in numbers:
            #     entities.add(number)
            #     neighbors[number].extend(number_columns + date_columns)
            #     for column_name in number_columns + date_columns:
            #         neighbors[column_name].append(number)
            #     entity_text[number] = number
            for entity, entity_neighbors in neighbors.items():
                neighbors[entity] = list(set(entity_neighbors))

            # Add "-1" as an entity only if we have date columns in the table because we will need
            # it as a wild-card in dates. The neighbors are the date columns.
            if "-1" not in neighbors and date_columns:
                entities.add("-1")
                neighbors["-1"] = date_columns
                entity_text["-1"] = "-1"
                for date_column in date_columns:
                    neighbors[date_column].append("-1")
            self._table_knowledge_graph = KnowledgeGraph(entities, dict(neighbors), entity_text)
        return self._table_knowledge_graph

    @classmethod
    def get_table_data_from_tagged_lines(
        cls, lines: List[List[str]]
    ) -> Tuple[List[Dict[str, Dict[str, str]]], Dict[str, Set[str]]]:
        column_index_to_name = {}
        header = lines[0]  # the first line is the header ("row\tcol\t...")
        index = 1
        table_data: List[Dict[str, Dict[str, str]]] = []
        while lines[index][0] == "-1":
            # column names start with fb:row.row.
            current_line = lines[index]
            column_name_sempre = current_line[2]
            column_index = int(current_line[1])
            column_name = column_name_sempre.replace("fb:row.row.", "")
            column_index_to_name[column_index] = column_name
            index += 1
        column_name_type_mapping: Dict[str, Set[str]] = defaultdict(set)
        last_row_index = -1
        for current_line in lines[1:]:
            row_index = int(current_line[0])
            if row_index == -1:
                continue  # header row
            column_index = int(current_line[1])
            if row_index != last_row_index:
                table_data.append({})
            node_info = dict(zip(header, current_line))
            cell_data: Dict[str, str] = {}
            column_name = column_index_to_name[column_index]
            if node_info["date"]:
                column_name_type_mapping[column_name].add("date")
                cell_data["date"] = node_info["date"]

            if node_info["number"]:
                column_name_type_mapping[column_name].add("number")
                cell_data["number"] = node_info["number"]

            if node_info["num2"]:
                column_name_type_mapping[column_name].add("num2")
                cell_data["num2"] = node_info["num2"]

            if node_info["content"] != "—":
                column_name_type_mapping[column_name].add("string")
                cell_data["string"] = node_info["content"]

            table_data[-1][column_name] = cell_data
            last_row_index = row_index

        return table_data, column_name_type_mapping

    @classmethod
    def get_table_data_from_untagged_lines(
        cls, lines: List[List[str]]
    ) -> Tuple[List[Dict[str, Dict[str, str]]], Dict[str, Set[str]]]:
        """
        This method will be called only when we do not have tagged information from CoreNLP. That is, when we are
        running the parser on data outside the WikiTableQuestions dataset. We try to do the same processing that
        CoreNLP does for WTQ, but what we do here may not be as effective.
        """
        table_data: List[Dict[str, Dict[str, str]]] = []
        column_index_to_name = {}
        column_names = lines[0]
        for column_index, column_name in enumerate(column_names):
            normalized_name = cls.normalize_string(column_name)
            column_index_to_name[column_index] = normalized_name

        column_name_type_mapping: Dict[str, Set[str]] = defaultdict(set)
        for row in lines[1:]:
            table_data.append({})
            for column_index, cell_value in enumerate(row):
                column_name = column_index_to_name[column_index]
                cell_data: Dict[str, str] = {}

                # Interpret the content as a date.
                try:
                    potential_date_string = str(Date.make_date(cell_value))
                    if potential_date_string != "-1":
                        # This means the string is a really a date.
                        cell_data["date"] = cell_value
                        column_name_type_mapping[column_name].add("date")
                except ValueError:
                    pass

                # Interpret the content as a number.
                try:
                    float(cell_value)
                    cell_data["number"] = cell_value
                    column_name_type_mapping[column_name].add("number")
                except ValueError:
                    pass

                # Interpret the content as a range or a score to get number and num2 out.
                if "-" in cell_value and len(cell_value.split("-")) == 2:
                    # This could be a number range or a score
                    cell_parts = cell_value.split("-")
                    try:
                        float(cell_parts[0])
                        float(cell_parts[1])
                        cell_data["number"] = cell_parts[0]
                        cell_data["num2"] = cell_parts[1]
                        column_name_type_mapping[column_name].add("number")
                        column_name_type_mapping[column_name].add("num2")
                    except ValueError:
                        pass

                # Interpret the content as a string.
                cell_data["string"] = cell_value
                column_name_type_mapping[column_name].add("string")
                table_data[-1][column_name] = cell_data

        return table_data, column_name_type_mapping

    @classmethod
    def read_from_lines(cls, lines: List) -> "TableQuestionContext":

        header = lines[0]
        if isinstance(header, list) and header[:6] == [
            "row",
            "col",
            "id",
            "content",
            "tokens",
            "lemmaTokens",
        ]:
            # These lines are from the tagged table file from the official dataset.
            table_data, column_name_type_mapping = cls.get_table_data_from_tagged_lines(lines)
        else:
            # We assume that the lines are just the table data, with rows being newline separated, and columns
            # being tab-separated.
            rows = [line.split("\t") for line in lines]  # type: ignore
            table_data, column_name_type_mapping = cls.get_table_data_from_untagged_lines(rows)
        # Each row is a mapping from column names to cell data. Cell data is a dict, where keys are
        # "string", "number", "num2" and "date", and the values are the corresponding values
        # extracted by CoreNLP.
        # Table data with each column split into different ones, depending on the types they have.
        table_data_with_column_types: List[Dict[str, CellValueType]] = []
        all_column_names: Set[str] = set()
        for table_row in table_data:
            table_data_with_column_types.append({})
            for column_name, cell_data in table_row.items():
                for column_type in column_name_type_mapping[column_name]:
                    typed_column_name = f"{column_type}_column:{column_name}"
                    all_column_names.add(typed_column_name)
                    cell_value_string = cell_data.get(column_type, None)
                    if column_type in ["number", "num2"]:
                        try:
                            cell_number = float(cell_value_string)
                        except (ValueError, TypeError):
                            cell_number = None
                        table_data_with_column_types[-1][typed_column_name] = cell_number
                    elif column_type == "date":
                        cell_date = None
                        if cell_value_string is not None:
                            cell_date = Date.make_date(cell_value_string)
                        table_data_with_column_types[-1][typed_column_name] = cell_date
                    else:
                        if cell_value_string is None:
                            normalized_string = None
                        else:
                            normalized_string = cls.normalize_string(cell_value_string)
                        table_data_with_column_types[-1][typed_column_name] = normalized_string
        return cls(
            table_data_with_column_types,
            column_name_type_mapping,
            all_column_names
        )

    @classmethod
    def read_from_file(cls, filename: str) -> "TableContext":
        with open(filename, "r") as file_pointer:
            reader = csv.reader(file_pointer, delimiter="\t", quoting=csv.QUOTE_NONE)
            lines = [line for line in reader]
            return cls.read_from_lines(lines)

    
    def _process_conjunction(self, entity_data):
        raise NotImplementedError

    @staticmethod
    def normalize_string(string: str) -> str:
        """
        These are the transformation rules used to normalize cell in column names in Sempre.  See
        ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
        ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
        rules here to normalize and canonicalize cells and columns in the same way so that we can
        match them against constants in logical forms appropriately.
        """
        # Normalization rules from Sempre
        # \u201A -> ,
        string = re.sub("‚", ",", string)
        string = re.sub("„", ",,", string)
        string = re.sub("[·・]", ".", string)
        string = re.sub("…", "...", string)
        string = re.sub("ˆ", "^", string)
        string = re.sub("˜", "~", string)
        string = re.sub("‹", "<", string)
        string = re.sub("›", ">", string)
        string = re.sub("[‘’´`]", "'", string)
        string = re.sub("[“”«»]", '"', string)
        string = re.sub("[•†‡²³]", "", string)
        string = re.sub("[‐‑–—−]", "-", string)
        # Oddly, some unicode characters get converted to _ instead of being stripped.  Not really
        # sure how sempre decides what to do with these...  TODO(mattg): can we just get rid of the
        # need for this function somehow?  It's causing a whole lot of headaches.
        string = re.sub("[ðø′″€⁄ªΣ]", "_", string)
        # This is such a mess.  There isn't just a block of unicode that we can strip out, because
        # sometimes sempre just strips diacritics...  We'll try stripping out a few separate
        # blocks, skipping the ones that sempre skips...
        string = re.sub("[\\u0180-\\u0210]", "", string).strip()
        string = re.sub("[\\u0220-\\uFFFF]", "", string).strip()
        string = string.replace("\\n", "_")
        string = re.sub("\\s+", " ", string)
        # Canonicalization rules from Sempre.
        string = re.sub("[^\\w]", "_", string)
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return unidecode(string.lower())
