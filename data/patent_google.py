import torch
import random
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, random_split
from pandas.io import gbq
import pandas as pd
import pickle
import re
import os

from tokenizer import TextTokenizer

from torchvision import transforms
from sklearn.model_selection import train_test_split


def pad_sequences(
    sequences, maxlen=None, dtype=torch.long, padding="pre", truncating="post", value=0
):
    """
    Pads sequences to the same length.

    Args:
    sequences (list of list of int): List of sequences to pad.
    maxlen (int, optional): Maximum length of all sequences. If None, uses the length of the longest sequence.
    dtype (torch.dtype, optional): Type of the output sequences.
    padding (str, optional): 'pre' or 'post', pad either before or after each sequence.
    truncating (str, optional): 'pre' or 'post', remove values from sequences larger than maxlen either in the beginning or in the end of the sequence.
    value (int, optional): Padding value.

    Returns:
    torch.Tensor: Padded sequences.
    """
    # Find the maximum length of the sequences
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    padded_sequences = []

    for seq in sequences:
        if len(seq) > maxlen:
            # Truncate sequences
            if truncating == "pre":
                truncated_seq = seq[-maxlen:]
            elif truncating == "post":
                truncated_seq = seq[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % truncating)
        else:
            truncated_seq = seq

        if len(truncated_seq) < maxlen:
            # Pad sequences
            if padding == "pre":
                padded_seq = [value] * (maxlen - len(truncated_seq)) + truncated_seq
            elif padding == "post":
                padded_seq = truncated_seq + [value] * (maxlen - len(truncated_seq))
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        else:
            padded_seq = truncated_seq

        padded_sequences.append(padded_seq)

    return torch.tensor(padded_sequences, dtype=dtype)


class PatentLandscapeExpander:
    """Class for L1&L2 expansion as 'Automated Patent Landscaping' describes.

    This object takes a seed set and a Google Cloud BigQuery project name and
    exposes methods for doing expansion of the project. The logical entry-point
    to the class is load_from_disk_or_do_expansion, which checks for cached
    expansions for the given self.seed_name, and if a previous run is available
    it will load it from disk and return it; otherwise, it does L1 and L2
    expansions, persists it in a cached 'data/[self.seed_name]/' directory,
    and returns the data to the caller.
    """

    seed_file = None
    # BigQuery must be enabled for this project
    bq_project = "patent-landscape-165715"
    patent_dataset = "patents-public-data:patents.publications_latest"
    # tmp_table = 'patents._tmp'
    l1_tmp_table = "patents._l1_tmp"
    l2_tmp_table = "patents._l2_tmp"
    antiseed_tmp_table = "patents.antiseed_tmp"
    country_codes = set(["US"])
    num_anti_seed_patents = 15000

    # ratios and multipler for finding uniquely common CPC codes from seed set
    min_ratio_of_code_to_seed = 0.04
    min_seed_multiplier = 50.0

    # persisted expansion information
    training_data_full_df = None
    seed_patents_df = None
    l1_patents_df = None
    l2_patents_df = None
    anti_seed_patents = None
    seed_data_path = None

    def __init__(
        self,
        seed_file,
        seed_name,
        bq_project=None,
        patent_dataset=None,
        num_antiseed=None,
    ):
        self.seed_file = seed_file
        self.seed_data_path = os.path.join("data", seed_name)

        if bq_project is not None:
            self.bq_project = bq_project
        if patent_dataset is not None:
            self.patent_dataset = patent_dataset
        # if tmp_table is not None:
        #    self.tmp_table = tmp_table
        if num_antiseed is not None:
            self.num_anti_seed_patents = num_antiseed

    def load_seeds_from_bq(self, seed_df):
        where_clause = ",".join("'" + seed_df.PubNum + "'")
        seed_patents_query = """
        SELECT
          b.publication_number,
          'Seed' as ExpansionLevel,
          STRING_AGG(citations.publication_number) AS refs,
          STRING_AGG(cpcs.code) AS cpc_codes
        FROM
          `patents-public-data.patents.publications` AS b,
          UNNEST(citation) AS citations,
          UNNEST(cpc) AS cpcs
        WHERE
        REGEXP_EXTRACT(b.publication_number, r'\w+-(\d+)-\w+') IN
        (
        {}
        )
        AND citations.publication_number != ''
        AND cpcs.code != ''
        GROUP BY b.publication_number
        LIMIT 1000
        ;
        """.format(
            where_clause
        )

        seed_patents_df = gbq.read_gbq(
            query=seed_patents_query,
            project_id=self.bq_project,
            # # verbose=False,
            dialect="standard",
        )

        return seed_patents_df

    def load_seed_pubs(self, seed_file=None):
        if seed_file is None:
            seed_file = self.seed_file

        seed_df = pd.read_csv(
            seed_file, header=None, names=["PubNum"], dtype={"PubNum": "str"}
        )

        return seed_df

    def bq_get_num_total_patents(self):
        num_patents_query = """
            SELECT
              COUNT(publication_number) AS num_patents
            FROM
              `patents-public-data.patents.publications` AS b
            WHERE
              country_code = 'US'
        """
        num_patents_df = gbq.read_gbq(
            query=num_patents_query,
            project_id=self.bq_project,
            # verbose=False,
            dialect="standard",
        )
        return num_patents_df

    def get_cpc_counts(self, seed_publications=None):
        where_clause = "1=1"
        if seed_publications is not None:
            where_clause = """
            REGEXP_EXTRACT(b.publication_number, r'\w+-(\d+)-\w+') IN
                (
                {}
                )
            """.format(
                ",".join("'" + seed_publications + "'")
            )

        cpc_counts_query = """
            SELECT
              cpcs.code,
              COUNT(cpcs.code) AS cpc_count
            FROM
              `patents-public-data.patents.publications` AS b,
              UNNEST(cpc) AS cpcs
            WHERE
            {}
            AND cpcs.code != ''
            AND country_code = 'US'
            GROUP BY cpcs.code
            ORDER BY cpc_count DESC;
            """.format(
            where_clause
        )

        return gbq.read_gbq(
            query=cpc_counts_query,
            project_id=self.bq_project,
            # # verbose=False,
            dialect="standard",
        )

    def compute_uniquely_common_cpc_codes_for_seed(self, seed_df):
        """
        Queries for CPC counts across all US patents and all Seed patents, then finds the CPC codes
        that are 50x more common in the Seed set than the rest of the patent corpus (and also appear in
        at least 5% of Seed patents). This then returns a Pandas dataframe of uniquely common codes
        as well as the table of CPC counts for reference. Note that this function makes several
        BigQuery queries on multi-terabyte datasets, so expect it to take a couple minutes.
        
        You should call this method like:
        uniquely_common_cpc_codes, cpc_counts_df = \
            expander.compute_uniquely_common_cpc_codes_for_seed(seed_df)
            
        where seed_df is the result of calling load_seed_pubs() in this class.
        """

        print("Querying for all US CPC Counts")
        us_cpc_counts_df = self.get_cpc_counts()
        print("Querying for Seed Set CPC Counts")
        seed_cpc_counts_df = self.get_cpc_counts(seed_df.PubNum)
        print("Querying to find total number of US patents")
        num_patents_df = self.bq_get_num_total_patents()

        num_seed_patents = seed_df.count().values[0]
        num_us_patents = num_patents_df["num_patents"].values[0]

        # Merge/join the dataframes on CPC code, suffixing them as appropriate
        cpc_counts_df = us_cpc_counts_df.merge(
            seed_cpc_counts_df, on="code", suffixes=("_us", "_seed")
        ).sort_values(ascending=False, by=["cpc_count_seed"])

        # For each CPC code, calculate the ratio of how often the code appears
        #  in the seed set vs the number of total seed patents
        cpc_counts_df["cpc_count_to_num_seeds_ratio"] = (
            cpc_counts_df.cpc_count_seed / num_seed_patents
        )
        # Similarly, calculate the ratio of CPC document frequencies vs total number of US patents
        cpc_counts_df["cpc_count_to_num_us_ratio"] = (
            cpc_counts_df.cpc_count_us / num_us_patents
        )
        # Calculate how much more frequently a CPC code occurs in the seed set vs full corpus of US patents
        cpc_counts_df["seed_relative_freq_ratio"] = (
            cpc_counts_df.cpc_count_to_num_seeds_ratio
            / cpc_counts_df.cpc_count_to_num_us_ratio
        )

        # We only care about codes that occur at least ~4% of the time in the seed set
        # AND are 50x more common in the seed set than the full corpus of US patents
        uniquely_common_cpc_codes = cpc_counts_df[
            (
                cpc_counts_df.cpc_count_to_num_seeds_ratio
                >= self.min_ratio_of_code_to_seed
            )
            & (cpc_counts_df.seed_relative_freq_ratio >= self.min_seed_multiplier)
        ]

        return uniquely_common_cpc_codes, cpc_counts_df

    def get_set_of_refs_filtered_by_country(self, seed_refs_series, country_codes):
        """
        Uses the refs column of the BigQuery on the seed set to compute the set of
        unique references out of the Seed set.
        """

        all_relevant_refs = set()
        for refs in seed_refs_series:
            for ref in refs.split(","):
                country_code = re.sub(r"(\w+)-(\w+)-\w+", r"\1", ref)
                if country_code in country_codes:
                    all_relevant_refs.add(ref)

        return all_relevant_refs

    # Expansion Functions
    def load_df_to_bq_tmp(self, df, tmp_table):
        """
        This function inserts the provided dataframe into a temp table in BigQuery, which
        is used in other parts of this class (e.g. L1 and L2 expansions) to join on by
        patent number.
        """
        print(
            "Loading dataframe with cols {}, shape {}, to {}".format(
                df.columns, df.shape, tmp_table
            )
        )
        gbq.to_gbq(
            dataframe=df,
            destination_table=tmp_table,
            project_id=self.bq_project,
            if_exists="replace",
            # verbose=False
        )

        print("Completed loading temp table.")

    def expand_l2(self, refs_series):
        self.load_df_to_bq_tmp(
            pd.DataFrame(refs_series, columns=["pub_num"]), self.l2_tmp_table
        )

        expansion_query = """
            SELECT
              b.publication_number,
              'L2' AS ExpansionLevel,
              STRING_AGG(citations.publication_number) AS refs
            FROM
              `patents-public-data.patents.publications` AS b,
              `{}` as tmp,
              UNNEST(citation) AS citations
            WHERE
            (
                b.publication_number = tmp.pub_num
            )
            AND citations.publication_number != ''
            GROUP BY b.publication_number
            LIMIT 1000
            ;
        """.format(
            self.l2_tmp_table
        )

        # print(expansion_query)
        expansion_df = gbq.read_gbq(
            query=expansion_query,
            project_id=self.bq_project,
            # # verbose=False,
            dialect="standard",
        )

        return expansion_df

    def expand_l1(self, cpc_codes_series, refs_series):
        self.load_df_to_bq_tmp(
            pd.DataFrame(refs_series, columns=["pub_num"]), self.l1_tmp_table
        )

        cpc_where_clause = ",".join("'" + cpc_codes_series + "'")

        expansion_query = """
            SELECT DISTINCT publication_number, ExpansionLevel, refs
            FROM
            (
            SELECT
              b.publication_number,
              'L1' as ExpansionLevel,
              STRING_AGG(citations.publication_number) AS refs
            FROM
              `patents-public-data.patents.publications` AS b,
              UNNEST(citation) AS citations,
              UNNEST(cpc) AS cpcs
            WHERE
            (
                cpcs.code IN
                (
                {}
                )
            )
            AND citations.publication_number != ''
            AND country_code IN ('US')
            GROUP BY b.publication_number

            UNION ALL

            SELECT
              b.publication_number,
              'L1' as ExpansionLevel,
              STRING_AGG(citations.publication_number) AS refs
            FROM
              `patents-public-data.patents.publications` AS b,
              `{}` as tmp,
              UNNEST(citation) AS citations
            WHERE
            (
                b.publication_number = tmp.pub_num
            )
            AND citations.publication_number != ''
            GROUP BY b.publication_number
            )
            LIMIT 1000
            ;
        """.format(
            cpc_where_clause, self.l1_tmp_table
        )

        # print(expansion_query)
        expansion_df = gbq.read_gbq(
            query=expansion_query,
            project_id=self.bq_project,
            # verbose=False,
            dialect="standard",
        )

        return expansion_df

    def anti_seed(self, seed_expansion_series):
        self.load_df_to_bq_tmp(
            pd.DataFrame(seed_expansion_series, columns=["pub_num"]),
            self.antiseed_tmp_table,
        )

        anti_seed_query = """
            SELECT DISTINCT
              b.publication_number,
              'AntiSeed' AS ExpansionLevel,
              rand() as random_num
            FROM
              `patents-public-data.patents.publications` AS b
            LEFT OUTER JOIN `{}` AS tmp ON b.publication_number = tmp.pub_num
            WHERE
            tmp.pub_num IS NULL
            AND country_code = 'US'
            ORDER BY random_num
            LIMIT {}
            # TODO: randomize results
            ;
        """.format(
            self.antiseed_tmp_table, self.num_anti_seed_patents
        )

        # print('Anti-seed query:\n{}'.format(anti_seed_query))
        anti_seed_df = gbq.read_gbq(
            query=anti_seed_query,
            project_id=self.bq_project,
            # verbose=False,
            dialect="standard",
        )

        return anti_seed_df

    def load_training_data_from_pubs(self, training_publications_df):
        tmp_table = "patents._tmp_training"
        self.load_df_to_bq_tmp(df=training_publications_df, tmp_table=tmp_table)

        # training_data_query = '''
        #     SELECT DISTINCT
        #         REGEXP_EXTRACT(LOWER(p.publication_number), r'[a-z]+-(\d+)-[a-z0-9]+') as pub_num,
        #         p.publication_number,
        #         p.family_id,
        #         p.priority_date,
        #         title.text as title_text,
        #         abstract.text as abstract_text,
        #         'unused' as claims_text,
        #         --SUBSTR(claims.text, 0, 5000) as claims_text,
        #         'unused' as description_text,
        #         --SUBSTR(description.text, 0, 5000) as description_text,
        #         STRING_AGG(citations.publication_number) AS refs,
        #         STRING_AGG(cpcs.code) AS cpcs
        #     FROM
        #       `patents-public-data.patents.publications` p,
        #       `{}` as tmp,
        #       UNNEST(p.title_localized) AS title,
        #       UNNEST(p.abstract_localized) AS abstract,
        #       UNNEST(p.claims_localized) AS claims,
        #       UNNEST(p.description_localized) AS description,
        #       UNNEST(p.title_localized) AS title_lang,
        #       UNNEST(p.abstract_localized) AS abstract_lang,
        #       UNNEST(p.claims_localized) AS claims_lang,
        #       UNNEST(p.description_localized) AS description_lang,
        #       UNNEST(citation) AS citations,
        #       UNNEST(cpc) AS cpcs
        #     WHERE
        #         p.publication_number = tmp.publication_number
        #         AND country_code = 'US'
        #         AND title_lang.language = 'en'
        #         AND abstract_lang.language = 'en'
        #         AND claims_lang.language = 'en'
        #         AND description_lang.language = 'en'
        #     GROUP BY p.publication_number, p.family_id, p.priority_date, title.text,
        #                 abstract.text, claims.text, description.text
        #     ;
        # '''.format(tmp_table)

        training_data_query = """
WITH filtered_publications AS (
    SELECT
        p.publication_number,
        p.family_id,
        p.priority_date,
        title.text AS title_text,
        -- abstract.text AS abstract_text,
        citations.publication_number AS citation_pub_number,
        cpcs.code AS cpc_code
    FROM
        `patents-public-data.patents.publications` p
    JOIN
        `{}` AS tmp
    ON
        p.publication_number = tmp.publication_number
    LEFT JOIN
        UNNEST(p.title_localized) AS title
    ON
        title.language = 'en'
    LEFT JOIN
        UNNEST(p.abstract_localized) AS abstract
    ON
        abstract.language = 'en'
    LEFT JOIN
        UNNEST(p.citation) AS citations
    LEFT JOIN
        UNNEST(p.cpc) AS cpcs
    WHERE
        p.country_code = 'US'
)

SELECT DISTINCT
    REGEXP_EXTRACT(LOWER(fp.publication_number), r'[a-z]+-(\d+)-[a-z0-9]+') AS pub_num,
    fp.publication_number,
    fp.family_id,
    fp.priority_date,
    fp.title_text,
    -- fp.abstract_text,
    'unused' AS claims_text,
    'unused' AS description_text,
    STRING_AGG(fp.citation_pub_number) AS refs,
    STRING_AGG(fp.cpc_code) AS cpcs
FROM
    filtered_publications fp
GROUP BY
    -- fp.publication_number, fp.family_id, fp.priority_date, fp.title_text, fp.abstract_text
    fp.publication_number, fp.family_id, fp.priority_date, fp.title_text
LIMIT 1000;
        """.format(
            tmp_table
        )

        print("Loading patent texts from provided publication numbers.")
        # print('Training data query:\n{}'.format(training_data_query))
        training_data_df = gbq.read_gbq(
            query=training_data_query,
            project_id=self.bq_project,
            # verbose=False,
            dialect="standard",
            configuration={
                "query": {"useQueryCache": True, "allowLargeResults": False}
            },
        )

        # 查询结果如果包含 abstract 会超出 bigquery 的限制，所以暂时不查询 abstract
        # 为了测试用，为结果表中的每一行添加一个随机的 abstract
        training_data_df["abstract_text"] = (
            "This is a test abstract for publication number "
            + training_data_df["publication_number"]
        )

        return training_data_df

    def do_full_expansion(self):
        """
        Does a full expansion on seed set as described in paper, using seed set
        to derive an anti-seed for use in supervised learning stage.
        
        Call this method like:
        seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = \
            expander.do_full_expansion(seed_file)
        """
        seed_df = self.load_seed_pubs(self.seed_file)

        seed_patents_df = self.load_seeds_from_bq(seed_df)

        # Level 1 Expansion
        ## getting unique seed CPC codes
        uniquely_common_cpc_codes, cpc_counts_df = (
            self.compute_uniquely_common_cpc_codes_for_seed(seed_df)
        )
        ## getting all the references out of the seed set
        all_relevant_refs = self.get_set_of_refs_filtered_by_country(
            seed_patents_df.refs, self.country_codes
        )
        print("Got {} relevant seed refs".format(len(all_relevant_refs)))
        ## actually doing expansion with CPC and references
        l1_patents_df = self.expand_l1(
            uniquely_common_cpc_codes.code, pd.Series(list(all_relevant_refs))
        )
        print("Shape of L1 expansion: {}".format(l1_patents_df.shape))

        # Level 2 Expansion
        l2_refs = self.get_set_of_refs_filtered_by_country(
            l1_patents_df.refs, self.country_codes
        )
        print("Got {} relevant L1->L2 refs".format(len(l2_refs)))
        l2_patents_df = self.expand_l2(pd.Series(list(l2_refs)))
        print("Shape of L2 expansion: {}".format(l2_patents_df.shape))

        # Get all publication numbers from Seed, L1, and L2
        ## for use in getting anti-seed
        # all_pub_nums = pd.Series(seed_patents_df.publication_number) \
        #     .append(l1_patents_df.publication_number) \
        #     .append(l2_patents_df.publication_number)
        all_pub_nums = pd.concat(
            [
                pd.Series(seed_patents_df.publication_number),
                l1_patents_df.publication_number,
                l2_patents_df.publication_number,
            ]
        )
        seed_and_expansion_pub_nums = set()
        for pub_num in all_pub_nums:
            seed_and_expansion_pub_nums.add(pub_num)
        print(
            "Size of union of [Seed, L1, and L2]: {}".format(
                len(seed_and_expansion_pub_nums)
            )
        )

        # get the anti-seed set!
        anti_seed_df = self.anti_seed(pd.Series(list(seed_and_expansion_pub_nums)))

        return seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_df

    def derive_training_data_from_seeds(self):
        """ """
        seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = (
            self.do_full_expansion()
        )
        # training_publications_df = \
        #     seed_patents_df.append(anti_seed_patents)[['publication_number', 'ExpansionLevel']]

        training_publications_df = pd.concat([seed_patents_df, anti_seed_patents])[
            ["publication_number", "ExpansionLevel"]
        ]

        print(
            "Loading training data text from {} publication numbers".format(
                training_publications_df.shape
            )
        )
        # 根据publication_number获取文本信息
        training_data_df = self.load_training_data_from_pubs(
            training_publications_df[["publication_number"]]
        )

        # training_data_full 包含了 seed 和 anti-seed 的数据
        print("Merging labels into training data.")
        training_data_full_df = training_data_df.merge(
            training_publications_df, on=["publication_number"]
        )

        return (
            training_data_full_df,
            seed_patents_df,
            l1_patents_df,
            l2_patents_df,
            anti_seed_patents,
        )

    def load_from_disk_or_do_expansion(self):
        """Loads data for seed from disk, else derives/persists, then returns it.

        Checks for cached expansions for the given self.seed_name, and if a
        previous run is available it will load it from disk and return it;
        otherwise, it does L1 and L2 expansions, persists it in a cached
        'data/[self.seed_name]/' directory, and returns the data to the caller.
        """

        landscape_data_path = os.path.join(self.seed_data_path, "landscape_data.pkl")

        if not os.path.exists(landscape_data_path):
            if not os.path.exists(self.seed_data_path):
                os.makedirs(self.seed_data_path)

            print("Loading landscape data from BigQuery.")
            (
                training_data_full_df,
                seed_patents_df,
                l1_patents_df,
                l2_patents_df,
                anti_seed_patents,
            ) = self.derive_training_data_from_seeds()

            print("Saving landscape data to {}.".format(landscape_data_path))
            with open(landscape_data_path, "wb") as outfile:
                pickle.dump(
                    (
                        training_data_full_df,
                        seed_patents_df,
                        l1_patents_df,
                        l2_patents_df,
                        anti_seed_patents,
                    ),
                    outfile,
                )
        else:
            print(
                "Loading landscape data from filesystem at {}".format(
                    landscape_data_path
                )
            )
            with open(landscape_data_path, "rb") as infile:

                landscape_data_deserialized = pickle.load(infile)

                (
                    training_data_full_df,
                    seed_patents_df,
                    l1_patents_df,
                    l2_patents_df,
                    anti_seed_patents,
                ) = landscape_data_deserialized

        self.training_data_full_df = training_data_full_df
        self.seed_patents_df = seed_patents_df
        self.l1_patents_df = l1_patents_df
        self.l2_patents_df = l2_patents_df
        self.anti_seed_patents = anti_seed_patents

        return (
            training_data_full_df,
            seed_patents_df,
            l1_patents_df,
            l2_patents_df,
            anti_seed_patents,
        )

    def sample_for_inference(self, train_data_util, sample_frac=0.20):
        if self.l1_patents_df is None:
            raise ValueError(
                "No patents loaded yet. Run expansion first (e.g., load_from_disc_or_do_expansion)"
            )

        inference_data_path = os.path.join(
            self.seed_data_path, "landscape_inference_data.pkl"
        )

        if not os.path.exists(inference_data_path):
            print("Loading inference data from BigQuery.")
            subset_l1_pub_nums = (
                self.l1_patents_df[["publication_number"]]
                .sample(frac=sample_frac)
                .reset_index(drop=True)
            )

            l1_texts = self.load_training_data_from_pubs(subset_l1_pub_nums)

            l1_subset = l1_texts[
                ["publication_number", "abstract_text", "refs", "cpcs"]
            ]

            # encode the data using the training data util
            padded_abstract_embeddings, refs_one_hot, cpc_one_hot = (
                train_data_util.prep_for_inference(
                    l1_subset.abstract_text, l1_subset.refs, l1_subset.cpcs
                )
            )

            print("Saving inference data to {}.".format(inference_data_path))
            with open(inference_data_path, "wb") as outfile:
                pickle.dump(
                    (
                        subset_l1_pub_nums,
                        l1_texts,
                        padded_abstract_embeddings,
                        refs_one_hot,
                        cpc_one_hot,
                    ),
                    outfile,
                )
        else:
            print(
                "Loading inference data from filesystem at {}".format(
                    inference_data_path
                )
            )
            with open(inference_data_path, "rb") as infile:
                inference_data_deserialized = pickle.load(infile)

                (
                    subset_l1_pub_nums,
                    l1_texts,
                    padded_abstract_embeddings,
                    refs_one_hot,
                    cpc_one_hot,
                ) = inference_data_deserialized

        return (
            subset_l1_pub_nums,
            l1_texts,
            padded_abstract_embeddings,
            refs_one_hot,
            cpc_one_hot,
        )


class PatentGoogle(data.Dataset):
    def __init__(self, w2v=None):
        super().__init__()
        self.prepare_data()
        self.w2v = w2v
        self.tokenizer = TextTokenizer()

        assert self.w2v is not None, "Word2Vec model is required for this dataset"

    def prepare_data(self):
        self.do_expansion()
        self.preprocess_data(50000, 500)

    def do_expansion(self):
        """
        从 bigQuery 中获得 expanded 后的数据 dataframe
        """
        bq_project = "patent-landscape-424113"
        seed_name = "video_codec"
        seed_file = "seeds/video_codec.seed.csv"
        patent_dataset = "patents-public-data:patents.publications_latest"
        num_anti_seed_patents = 1500

        expander = PatentLandscapeExpander(
            seed_file,
            seed_name,
            bq_project=bq_project,
            patent_dataset=patent_dataset,
            num_antiseed=num_anti_seed_patents,
        )

        (
            training_data_full_df,
            seed_patents_df,
            l1_patents_df,
            l2_patents_df,
            anti_seed_patents,
        ) = expander.load_from_disk_or_do_expansion()
        print(training_data_full_df.head(5))

        self.data = training_data_full_df[
            [
                "publication_number",
                "title_text",
                "abstract_text",
                "claims_text",
                "description_text",
                "ExpansionLevel",
                "refs",
                "cpcs",
            ]
        ]

        print("Seed/Positive examples:")
        print(self.data[self.data.ExpansionLevel == "Seed"].count())

        print("\n\nAnti-Seed/Negative examples:")
        print(self.data[self.data.ExpansionLevel == "AntiSeed"].count())

    def preprocess_data(
        self,
        refs_vocab_size,
        cpc_vocab_size,
    ):
        """
        对数据进行预处理
        """
        labels_series, text_series, refs_series, cpcs_series = (
            self.data.ExpansionLevel,
            self.data.abstract_text,
            self.data.refs,
            self.data.cpcs,
        )
        refs_series.fillna("", inplace=True)
        cpcs_series.fillna("", inplace=True)

        self.text_embeddings = self.text_series_to_embeddings(text_series)
        self.refs_tokenizer, self.refs_one_hot = self.tokenize_to_onehot_matrix(
            refs_series, refs_vocab_size
        )
        self.cpc_tokenizer, self.cpc_one_hot = self.tokenize_to_onehot_matrix(
            cpcs_series, cpc_vocab_size
        )
        self.labels = self.label_series_to_index(labels_series)

        doc_lengths = list(map(len, self.text_embeddings))
        median_doc_length = int(np.median(doc_lengths))
        max_doc_length = int(np.max(doc_lengths))
        print(
            f"doc lengths for embeddings: median:{median_doc_length}, mean:{np.mean(doc_lengths)}, max:{max_doc_length}"
        )

        self.sequence_length = max_doc_length

        print("Sequence length: ", self.sequence_length)

        self.text_embeddings = pad_sequences(
            self.text_embeddings,
            maxlen=self.sequence_length,
            padding="pre",
            truncating="post",
        )

    def __getitem__(self, idx):
        return (
            self.text_embeddings[idx],
            self.refs_one_hot[idx],
            self.cpc_one_hot[idx],
            self.labels[idx],
        )

    def __len__(self):
        return len(self.labels)

    def tokenize_to_onehot_matrix(self, text_series, vocab_size):
        """处理 cpcs 和 refs 到 onehot matrix"""
        from tokenizer import OneHotTokenizer

        one_hot_tokenizer = OneHotTokenizer(
            num_words=vocab_size,
            split=",",
            filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n',
            lower=False,
        )
        one_hot_tokenizer.fit_on_texts(text_series)
        one_hot_tokenizer.index_word = {
            idx: word for word, idx in one_hot_tokenizer.word_index.items()
        }
        text_one_hot = one_hot_tokenizer.texts_to_matrix(text_series)
        return one_hot_tokenizer, text_one_hot

    def text_series_to_embeddings(self, raw_series_text):
        """
        Takes as input a series of text and associated labels
        """
        tokenized_text = self.tokenizer.tokenize_series(raw_series_text)
        word_to_index_dict = self.w2v.word_to_index
        tokenized_indexed_text = []

        for idx in range(0, len(tokenized_text)):
            text = tokenized_text[idx]
            text_word_indexes = []
            for word in text:
                if word in word_to_index_dict:
                    word_idx = word_to_index_dict[word]
                else:
                    word_idx = -1
                    # word_idx = word_to_index_dict['<unk>']
                # this skips 'the' so it can be used for dynamic rnn
                if word_idx > 0:
                    text_word_indexes.append(word_idx)

            tokenized_indexed_text.append(text_word_indexes)

        return tokenized_indexed_text

    def label_series_to_index(self, labels_series):
        labels_indexed = []
        for idx in range(0, len(labels_series)):
            label = labels_series[idx]
            # 'tokenize' on the label is basically normalizing it
            tokenized_label = self.tokenizer.tokenize(label)[0]
            label_idx = 1 if tokenized_label == "antiseed" else 0
            labels_indexed.append(label_idx)
        return labels_indexed
