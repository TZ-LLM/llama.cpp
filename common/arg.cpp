#include "arg.h"

#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <climits>
#include <cstdarg>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <iostream>

#include "json-schema-to-grammar.h"

using json = nlohmann::ordered_json;

llama_arg & llama_arg::set_examples(std::initializer_list<enum llama_example> examples) {
    this->examples = std::move(examples);
    return *this;
}

llama_arg & llama_arg::set_env(const char * env) {
    help = help + "\n(env: " + env + ")";
    this->env = env;
    return *this;
}

llama_arg & llama_arg::set_sparam() {
    is_sparam = true;
    return *this;
}

bool llama_arg::in_example(enum llama_example ex) {
    return examples.find(ex) != examples.end();
}

bool llama_arg::get_value_from_env(std::string & output) {
    if (env == nullptr) return false;
    char * value = std::getenv(env);
    if (value) {
        output = value;
        return true;
    }
    return false;
}

bool llama_arg::has_value_from_env() {
    return env != nullptr && std::getenv(env);
}

static std::vector<std::string> break_str_into_lines(std::string input, size_t max_char_per_line) {
    std::vector<std::string> result;
    std::istringstream iss(input);
    std::string line;
    auto add_line = [&](const std::string& l) {
        if (l.length() <= max_char_per_line) {
            result.push_back(l);
        } else {
            std::istringstream line_stream(l);
            std::string word, current_line;
            while (line_stream >> word) {
                if (current_line.length() + !current_line.empty() + word.length() > max_char_per_line) {
                    if (!current_line.empty()) result.push_back(current_line);
                    current_line = word;
                } else {
                    current_line += (!current_line.empty() ? " " : "") + word;
                }
            }
            if (!current_line.empty()) result.push_back(current_line);
        }
    };
    while (std::getline(iss, line)) {
        add_line(line);
    }
    return result;
}

std::string llama_arg::to_string() {
    // params for printing to console
    const static int n_leading_spaces = 40;
    const static int n_char_per_line_help = 70; // TODO: detect this based on current console
    std::string leading_spaces(n_leading_spaces, ' ');

    std::ostringstream ss;
    for (const auto arg : args) {
        if (arg == args.front()) {
            if (args.size() == 1) {
                ss << arg;
            } else {
                // first arg is usually abbreviation, we need padding to make it more beautiful
                auto tmp = std::string(arg) + ", ";
                auto spaces = std::string(std::max(0, 7 - (int)tmp.size()), ' ');
                ss << tmp << spaces;
            }
        } else {
            ss << arg << (arg != args.back() ? ", " : "");
        }
    }
    if (value_hint) ss << " " << value_hint;
    if (value_hint_2) ss << " " << value_hint_2;
    if (ss.tellp() > n_leading_spaces - 3) {
        // current line is too long, add new line
        ss << "\n" << leading_spaces;
    } else {
        // padding between arg and help, same line
        ss << std::string(leading_spaces.size() - ss.tellp(), ' ');
    }
    const auto help_lines = break_str_into_lines(help, n_char_per_line_help);
    for (const auto & line : help_lines) {
        ss << (&line == &help_lines.front() ? "" : leading_spaces) << line << "\n";
    }
    return ss.str();
}

//
// utils
//

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

static void gpt_params_handle_model_default(gpt_params & params) {
    if (!params.hf_repo.empty()) {
        // short-hand to avoid specifying --hf-file -> default it to --model
        if (params.hf_file.empty()) {
            if (params.model.empty()) {
                throw std::invalid_argument("error: --hf-repo requires either --hf-file or --model\n");
            }
            params.hf_file = params.model;
        } else if (params.model.empty()) {
            params.model = fs_get_cache_file(string_split(params.hf_file, '/').back());
        }
    } else if (!params.model_url.empty()) {
        if (params.model.empty()) {
            auto f = string_split(params.model_url, '#').front();
            f = string_split(f, '?').front();
            params.model = fs_get_cache_file(string_split(f, '/').back());
        }
    } else if (params.model.empty()) {
        params.model = DEFAULT_MODEL_PATH;
    }
}

//
// CLI argument parsing functions
//

static bool gpt_params_parse_ex(int argc, char ** argv, gpt_params_context & ctx_arg) {
    std::string arg;
    const std::string arg_prefix = "--";
    gpt_params & params = ctx_arg.params;

    std::unordered_map<std::string, llama_arg *> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = &opt;
        }
    }
    std::cout << "In function gpt_params_parse_ex" << std::endl;
    // handle environment variables
    for (auto & opt : ctx_arg.options) {
        std::string value;
        if (opt.get_value_from_env(value)) {
            try {
                if (opt.handler_void && (value == "1" || value == "true")) {
                    opt.handler_void(params);
                }
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(value));
                }
                if (opt.handler_string) {
                    opt.handler_string(params, value);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(format(
                    "error while handling environment variable \"%s\": %s\n\n", opt.env, e.what()));
            }
        }
    }
    std::cout << "After handling environment variables" << std::endl;
    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };
    std::cout << "Before for loop" << std::endl;
    for (int i = 1; i < argc; i++) {
        const std::string arg_prefix = "--";

        std::string arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (arg_to_options.find(arg) == arg_to_options.end()) {
            throw std::invalid_argument(format("error: invalid argument: %s", arg.c_str()));
        }
        auto opt = *arg_to_options[arg];
        if (opt.has_value_from_env()) {
            fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n", opt.env, arg.c_str());
        }
        try {
            if (opt.handler_void) {
                opt.handler_void(params);
                continue;
            }

            // arg with single value
            check_arg(i);
            std::string val = argv[++i];
            if (opt.handler_int) {
                opt.handler_int(params, std::stoi(val));
                continue;
            }
            if (opt.handler_string) {
                opt.handler_string(params, val);
                continue;
            }

            // arg with 2 values
            check_arg(i);
            std::string val2 = argv[++i];
            if (opt.handler_str_str) {
                opt.handler_str_str(params, val, val2);
                continue;
            }
        } catch (std::exception & e) {
            throw std::invalid_argument(format(
                "error while handling argument \"%s\": %s\n\n"
                "usage:\n%s\n\nto show complete usage, run with -h",
                arg.c_str(), e.what(), arg_to_options[arg]->to_string().c_str()));
        }
    }
    std::cout << "After for loop" << std::endl;
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    postprocess_cpu_params(params.draft_cpuparams, &params.cpuparams);
    postprocess_cpu_params(params.draft_cpuparams_batch, &params.cpuparams_batch);

    if (params.prompt_cache_all && (params.interactive || params.interactive_first)) {
        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }
    std::cout << "After postprocess_cpu_params" << std::endl;
    gpt_params_handle_model_default(params);

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
    }
    std::cout<<"After gpt_params_handle_model_default"<<std::endl;
    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }
    std::cout<<"After kv_overrides"<<std::endl;
    return true;
}

static void gpt_params_print_usage(gpt_params_context & ctx_arg) {
    auto print_options = [](std::vector<llama_arg *> & options) {
        for (llama_arg * opt : options) {
            printf("%s", opt->to_string().c_str());
        }
    };

    std::cout << "In function gpt_params_print_usage" << std::endl;

    std::vector<llama_arg *> common_options;
    std::vector<llama_arg *> sparam_options;
    std::vector<llama_arg *> specific_options;
    for (auto & opt : ctx_arg.options) {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.is_sparam) {
            sparam_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }
    printf("----- common params -----\n\n");
    print_options(common_options);
    printf("\n\n----- sampling params -----\n\n");
    print_options(sparam_options);
    // TODO: maybe convert enum llama_example to string
    printf("\n\n----- example-specific params -----\n\n");
    print_options(specific_options);
}

const char *ultra_chat_prompts[] = {
    "These instructions apply to section-based themes (Responsive 6.0+, Retina 4.0+, Parallax 3.0+ Turbo 2.0+, Mobilia 5.0+). What theme version am I using?\nOn your Collections pages & Featured Collections sections, you can easily show the secondary image of a product on hover by enabling one of the theme's built-in settings!\nYour Collection pages & Featured Collections sections will now display the secondary product image just by hovering over that product image thumbnail.\nDoes this feature apply to all sections of the theme or just specific ones as listed in the text material?",
    "Which famous landmarks should I visit in London, beyond the usual ones?",
    "Write a comprehensive blog post of at least 1000 words about the top 10 most eco-friendly cities in the world and their renewable energy initiatives. Use a formal and informative tone, and include statistics, case studies, and expert opinions to support your claims. Make sure to cover various aspects of sustainability, such as public transportation, waste management, green buildings, and carbon emissions reduction. Additionally, provide specific examples of innovative renewable energy projects implemented in each city and assess their impact on the environment and the local community. Provide credible sources and links to relevant articles, reports, and websites to enhance the article's credibility and appeal to the readers.",
    "De León, previewing the speech he will give today, said he will highlight his Senate Bill 535, which directs a quarter of the proceeds from the Greenhouse Gas Reduction Fund to projects that benefit disadvantaged communities.\nOn Thursday, de León nodded approvingly as a string of leading scientists and religious leaders gathered for hours of weedy policy discussions on the impacts of climate change, including gloomy predictions on mortality attributable to air pollution.\nSEIU HEADS TO THE BAR: Employees of the State Bar of California represented by SEIU are planning a picket line today at the bar building in Los Angeles to protest the latest contract offer. What is the reason for SEIU employees planning a picket line at the State Bar of California building in Los Angeles?",
    "Write an essay that evaluates the positive and negative influence of social media on personal relationships, citing specific examples and research to support your claims. Analyze the ways in which social media usage affects communication patterns, trust levels, and emotional connections in romantic partnerships, friendships, and family relationships. Consider the role of social comparison, cyberbullying, and privacy concerns in shaping the impact of social media use on personal relationships. Finally, provide recommendations for how individuals can cultivate healthy and meaningful relationships in the age of social media.",
    "Write a Python program that can identify and remove duplicate images from a given folder or directory. The program should be able to compare images based on their content and file size, and determine which ones are exact duplicates. The duplicates should then be deleted or moved to a separate folder. The program should also provide a user-friendly interface with clear instructions on how to use it.",
    "Please provide a specific cooking method for the salmon, as well as any seasonings or ingredients necessary to complete the dish. Additionally, include instructions for how to properly prepare the salmon before cooking, such as removing the skin or deboning. Please write in step-by-step format, using clear and concise language.",
    "What are the classic cocktails that every bartender should know how to make and what are their ingredients?",
    "Write step-by-step instructions detailing how to make a creamy and tangy vegan cashew cheese, including a list of required ingredients and suggested serving methods. Include variations and suggestions for different flavors and textures.",
    "How did the Spanish conquest of the Low Countries in the 16th century impact European politics and culture?",

    // "What are the rules and restrictions in place for COVID-19 in the city?",
    // "Implement a queue data structure in Java that can enqueue and dequeue elements. The queue should have a maximum size of n and should be implemented using an array. The enqueue operation should add elements to the back of the queue, and the dequeue operation should remove elements from the front of the queue. If the queue is full, the enqueue operation should throw an exception, and if the queue is empty, the dequeue operation should return null. The queue should also have a method to check if it is empty and a method to get the size of the queue. Finally, the implementation should use generics to allow storing of any data type in the queue.",
    // "Write step-by-step instructions for making a traditional snowman using three large snowballs, including information on how to properly pack the snow, tools (if applicable), and recommended size ratios for each snowball. Be sure to include any additional details on decorations or accessories, if desired.",
    // "Provide a step-by-step recipe on how to make a rich and flavorful tomato soup from scratch, including how to choose and prepare the ingredients, the proper order of steps, cooking times, and recommended garnishes or toppings to enhance the taste and presentation.",
    // "Using at least 3 scholarly articles, write a comprehensive comparative analysis between two employee wellness programs and their effects on productivity in the workplace. Include a detailed description of the programs, their implementation, and any observed changes in productivity metrics. Also, discuss the limitations and potential areas for improvement in both programs. Your analysis should be at least 2,500 words and follow APA formatting style.",
    // "Is it possible to have a combination of drawers, cupboards, and wine racks in the same sideboard?: Hand Made any size, any combination or drawers, cupboards, shelving, glazed, solid doors, wine racks.\nStunning Oak top shown here but can be made with pine, Reclaimed Timber, Driftwood, Rustic Plank and beech tops also.\n4 Door Oslo sideboard as shown from £815. Can be made any size with your combination of drawers, cupboards, wine racks, shelving, door and plinth styles available, choice of different worktops (shown here with 20mm bull nose oak top) and finished in any Farrow and Ball colour of your choice.\nn.b prices are correct as of 01-05/16 but may be subject to change – please contact us for a quote based on your requirements, sizes and needs.",
    // "What are the different CoPs issued by the Canadian Sport Parachute Association, and how can skydivers progress to the higher levels? Answer according to: After completing Accelerated Freefall (AFF) and acquiring your Solo Certificate of Proficiency (CoP), you’ll be able to do solo skydives at skydive centers in Canada. But, your journey into the sport of skydiving has just begun. You essentially have just earned a license to learn.\nEdmonton Skydive is a member of the Canadian Sport Parachute Association (CSPA). Our instructors and coaches help train skydivers to progress their skills and earn their respective CSPA licenses and privileges. The CSPA issues Solo, “A”, “B”, “C”, and “D” CoPs.\nWe are a full time and full service skydive center. We have coaches and instructors onsite always to help you progress to the next level or acquire your next CoP.\nOur team organizes Intermediate Skills Camps once a month to provide up and coming jumpers with free coaching to help master specific skills or to finalize a few outstanding requirements for your next CoP.\nFor more experienced and advanced flyers, we regularly offer freefly, wingsuit or big way load organizing. Please check our events for dates on Intermediate Skills Camps or Load Organizing.",
    // "Write a descriptive narrative about a specific moment in your life when you felt truly happy and content. Use sensory details, metaphor, and dialogue to illustrate the scene, setting, and emotions. Be sure to reflect on how this moment impacted you and whether it changed your perspective on life.",
    // "Can you summarize the key objective of the EAA TC-RBA working group mentioned in the given text material? Answer according to: A major task of the working group will be to try to find consensus on one single concept for airborne sound insulation and one single concept for impact sound to be used in the future. Today there are a large variety of different concepts being used in Europe. This EAA TC-RBA working group might play an important role by recommending specific concepts.\nThe working group can work electronically and arrange specific meetings or workshops. The organisation of structured sessions in future conferences will also be considered.",
    // "How have technological advancements impacted agriculture and economic growth in Tripura?",
};

const char *persona_u1[] = {
    "I am 32.\nI do not want a job.\nI play video games all day.\nI still live at home with my parents.",
    "I write.\nI work at mcdonald s.\nI watch youtube videos.\nI have brown hair.",
    "I am bald.\nI like to swim.\nMy favorite drink is chocolate milk.\nI don t work.",
    "I am a man.\nI work on trucks.\nI have a doberman.\nI am twenty five years old.\nMy favorite food is pizza.",
    "I ve a dog.\nMy favorite color is red.\nI m not very tall.\nI ve two siblings.",
    // "I am married.\nI love to travel.\nI have a passion for god.\nMy favorite book is the bible.\nI am a older gentlemen.",
    // "I live alone.\nI like to play video games.\nI play guitar on my free time.\nI am an amateur chef who cooks 5 different cuisines.",
    // "I am male.\nI like potatoes.\nI work in accounting.\nI lease my car.",
    // "I am a vegan.\nI have a cat named jasper.\nMy favorite flower is a rose.\nI work as a barista.\nMy favorite color is orange.",
    // "I have a cat.\nMy mother was a nurse.\nI like to be alone.\nI wait tables at a cafe.\nI love to sew.",
};
const char *persona_u2[] = {
    "My favorite drink is iced coffee.\nI have a black belt in karate.\nI m in a jazz band and play the saxophone.\nI vacation along lake michigan every summer.",
    "I want to move.\nI don t like feeling controlled.\nI wish i could take back a mistake.\nI have a harsh inner critic.\nI don t like my reputation.",
    "My favorite store is american eagle.\nI enjoy spending time with my family.\nI love to go running.\nI do not like vegetables.",
    "I listen to katie perry while in the shower.\nI m a breath of fresh air.\nI drive a yellow convertible in the rain.\nI work as a lifeguard at a beach.",
    "I love being in nature.\nMy mother is a teacher.\nI can speak three languages.\nI like to bike.",
    // "I am a proofreader for a greeting card company.\nI love to read.\nMy favorite movie is goodfellas.\nMy parents have been married for 40 years.",
    // "I enjoy my job , as a teacher.\nI live in a big city.\nI enjoy video games.\nMagic mike is my favorite movie.\nI enjoy singing.",
    // "I am 10 years old.\nI like to watch tv.\nMy name is arnold.\nI love ham and cheese sandwiches.",
    // "I sleep 8 hours.\nI use the computer all day.\nMy mother looks after my grandmother.\nI prefer texting over phone calls.",
    // "I like to cook.\nI worked as a nurse for many years.\nMy husband was a salesman.\nI have lived in several different states.",

};
const char *persona_c[] = {
    "User 1: Hi! I'm [user 1's name].\nUser 2: Hi [user 1's name], I'm [user 2's name].\nUser 1: What do you do for fun?\nUser 2: I like to play video games, go to the beach, and read.\nUser 1: I like to play video games too! I'm not much of a reader, though.\nUser 2: What video games do you like to play?\nUser 1: I like to play a lot of different games, but I'm really into competitive online games right now.\nUser 2: I'm not really into competitive games, I like to play more relaxing games.\nUser 1: That's cool. What kind of relaxing games do you like to play?\nUser 2: I like to play puzzle games, simulation games, and story-based games.\nUser 1: I've never been much of a puzzle game person, but I do like simulation games and story-based games.\nUser 2: Nice! What's your favorite simulation game?\nUser 1: I like Stardew Valley a lot. It's a farming game, but it's also really relaxing and fun.\nUser 2: I've heard good things about that game. I might have to check it out.\nUser 1: You should! It's a lot of fun.\nUser 2: Well, I'm glad we met. Maybe we can play some games together sometime.\nUser 1: That would be fun!\nUser 2: Great! I'll send you my Steam name.\nUser 1: Ok, sounds good.",
    "User 1: Hi!\nUser 2: Hey!\nUser 1: What's up?\nUser 2: Not much. Just hanging out.\nUser 1: Cool. What do you like to do for fun?\nUser 2: I like to watch YouTube videos, go to concerts, and read.\nUser 1: Nice. I like to write, watch YouTube videos, and listen to music.\nUser 2: What do you write?\nUser 1: I write poetry, short stories, and essays.\nUser 2: That's cool. I've always wanted to write a book.\nUser 1: You should try! It's a lot of fun.\nUser 2: I'm not sure if I'm good enough.\nUser 1: You don't have to be good enough. Just start writing and see what happens.\nUser 2: I'll think about it.\nUser 1: Cool. So, what do you want to do after we graduate?\nUser 2: I want to move to a new city. I'm tired of living in this one.\nUser 1: I know what you mean. I'm also ready for a change.\nUser 2: Maybe we can move to the same city.\nUser 1: That would be cool! We could be roommates.\nUser 2: That would be fun!\nUser 1: So, what are we waiting for? Let's go pack our bags!\nUser 2: Haha, okay! I'm ready!",
    "User 1: Hello!\nUser 2: Hi!\nUser 1: What do you like to do for fun?\nUser 2: I love to spend time with my family, go running, and shop.\nUser 1: Oh, that's great! I love to swim and go shopping too.\nUser 2: That's awesome! We should go shopping sometime.\nUser 1: That would be fun! What's your favorite store?\nUser 2: My favorite store is American Eagle.\nUser 1: Oh, I love that store! I always find great clothes there.\nUser 2: Me too! They have a great selection of clothes.\nUser 1: What kind of clothes do you like to wear?\nUser 2: I like to wear casual clothes, like jeans and t-shirts.\nUser 1: Me too! I love how comfortable they are.\nUser 2: Me too! They're perfect for running.\nUser 1: I know, right? I love running. It's so relaxing.\nUser 2: Me too! It's a great way to get in shape.\nUser 1: I know, right? I love how I feel after I've run.\nUser 2: Me too! It's a great way to start your day.\nUser 1: I know, right? I love waking up early and going for a run.\nUser 2: Me too! It's the best way to start the day.",
    "User 1: What do you do for work?\nUser 2: I'm a lifeguard at a beach.\nUser 1: Cool! I've always wanted to be a lifeguard.\nUser 2: It's a lot of fun, but it can be dangerous.\nUser 1: I bet. Do you ever have to save anyone?\nUser 2: A few times, yeah.\nUser 1: What's it like to save someone's life?\nUser 2: It's a really rewarding feeling.\nUser 1: I can imagine. So, what's your favorite part of the job?\nUser 2: The best part is getting to help people.\nUser 1: That's a good answer. What's your least favorite part of the job?\nUser 2: The worst part is the long hours.\nUser 1: I can imagine. Do you have any interesting stories from your time as a lifeguard?\nUser 2: Yeah, I have a few.\nUser 1: Like what?\nUser 2: Well, one time, I had to save a kid who was drowning.\nUser 1: Oh no! What happened?\nUser 2: He was playing in the ocean and he got caught in a riptide.\nUser 1: That's scary. How did you save him?\nUser 2: I swam out to him and pulled him to shore.\nUser 1: Wow, that's amazing! You're a hero!\nUser 2: Thanks, but I was just doing my job.\nUser 1: You saved someone's life. That's more than just your job. That's heroic.\nUser 2: Thanks. It's good to hear that you appreciate it.",
    "User 1: Hi, I'm John!\nUser 2: Hi, nice to meet you, I'm Jane.\nUser 1: What do you like to do for fun?\nUser 2: I love being outdoors, hiking and biking are my favorite activities.\nUser 1: I love hiking too! Do you like to go with friends or by yourself?\nUser 2: I usually go hiking with friends, but I've also done some solo hikes.\nUser 1: I've been wanting to try a solo hike, but I'm not sure if I'm brave enough.\nUser 2: It's definitely a good idea to start with a shorter hike if you're not used to hiking alone.\nUser 1: That's a good idea. I'll start with a short hike and see how I feel.\nUser 2: Sounds like a plan. What's your favorite color?\nUser 1: My favorite color is red.\nUser 2: Red is a good color. I like red, too.\nUser 1: I also have a dog. His name is Max.\nUser 2: I love dogs! What kind of dog is Max?\nUser 1: Max is a golden retriever. He's the sweetest dog ever.\nUser 2: Golden retrievers are the best! I had a golden retriever when I was growing up.\nUser 1: They're such amazing dogs. I'm so glad I have Max in my life.\nUser 2: It sounds like you have a great dog.",
    // "User 1: Hello there, what do you do for a living?\nUser 2: I am a proofreader for a greeting card company.\nUser 1: Oh, that sounds interesting. Do you enjoy it?\nUser 2: I do, it's fun to work with words and help make sure they're correct.\nUser 1: That's great. What are some of your favorite books?\nUser 2: I love to read, but my favorite book is Goodfellas.\nUser 1: I've never heard of that one. What's it about?\nUser 2: It's a true story about a group of gangsters in the 1950s. It's very well done, and it's a classic.\nUser 1: Sounds interesting, I'll have to check it out.\nUser 2: You won't be disappointed.\nUser 1: I'm married, and my favorite book is the Bible. I have a passion for God, and love to travel.\nUser 2: That's great. I've always wanted to travel, but I've never had the chance.\nUser 1: You should definitely go! There's so much to see and experience in the world.\nUser 2: I know, I've heard great things. Maybe I'll take a trip this year.\nUser 1: That would be great!\nUser 2: Thanks for the recommendation.\nUser 1: No problem.",
    // "User 1: What do you like to do for fun?\nUser 2: I enjoy playing video games and singing.\nUser 1: Cool! I like to play video games too. What kind of games do you like to play?\nUser 2: I like to play all kinds of games, but my favorite genre is probably action RPGs.\nUser 1: I love action RPGs too! What are some of your favorite games?\nUser 2: I love the Witcher series, the Dark Souls series, and the Mass Effect series.\nUser 1: Those are all great games! I've played all of them except for the Mass Effect series.\nUser 2: You should definitely play the Mass Effect series. It's one of the best sci-fi games ever made.\nUser 1: I'll check it out. What about you? What do you like to do for fun?\nUser 2: I like to sing. I'm in a choir and I also like to sing karaoke.\nUser 1: Oh cool! I love to sing too. I used to be in a choir when I was younger.\nUser 2: Me too! I was in a choir when I was in high school.\nUser 1: What kind of music do you like to sing?\nUser 2: I like to sing all kinds of music, but my favorite genre is probably pop.\nUser 1: I love pop music too! I'm a big fan of Taylor Swift.\nUser 2: Me too! I love Taylor Swift. She's so talented.\nUser 1: She is! I'm excited for her new album.\nUser 2: Me too! I'm sure it's going to be great.",
    // "User 1: What do you like to do for fun?\nUser 2: I like to watch TV and play games.\nUser 1: What kind of games do you like to play?\nUser 2: I like to play video games and board games.\nUser 1: I like video games too! What are your favorite games?\nUser 2: I like to play Mario Kart and Minecraft.\nUser 1: I love Mario Kart! I'm not very good at Minecraft, but I still enjoy playing it.\nUser 2: I'm not very good at Mario Kart, but I still enjoy playing it too.\nUser 1: What else do you like to do for fun?\nUser 2: I like to go to the park and play with my friends.\nUser 1: That sounds like fun! I like to go to the park too.\nUser 2: I also like to go to the movies.\nUser 1: I like to go to the movies too! What was the last movie you saw?\nUser 2: The last movie I saw was Frozen 2.\nUser 1: I loved Frozen 2! I saw it twice in theaters.\nUser 2: I thought it was really good too! I'm excited for the next one.\nUser 1: Me too! I hope it's as good as the first one.\nUser 2: Me too!\nUser 1: What are your favorite foods?\nUser 2: I like ham and cheese sandwiches.\nUser 1: I love ham and cheese sandwiches! I used to eat them all the time when I was a kid.\nUser 2: I still eat them all the time! They're so good.\nUser 1: They are! I'm glad we have something in common.",
    // "User 1: Hi there!\nUser 2: Hello!\nUser 1: What do you do for a living?\nUser 2: I'm a software engineer.\nUser 1: Oh, that sounds interesting! What kind of software do you work on?\nUser 2: I work on a lot of different types of software, but my favorite is working on games.\nUser 1: That's cool! I love video games. What games have you worked on?\nUser 2: I've worked on a lot of different games, but my favorite was one called \"The Sims\".\nUser 1: Oh, I love \"The Sims\"! I've played it for years.\nUser 2: Me too! It's such a fun game.\nUser 1: What's your favorite thing about \"The Sims\"?\nUser 2: I love being able to create my own characters and build my own houses.\nUser 1: Me too! It's so much fun to make your own world.\nUser 2: I also love the different ways you can play the game. You can just build houses and create characters, or you can play the game like a role-playing game.\nUser 1: I've never tried playing it like a role-playing game. I'll have to give that a try sometime.\nUser 2: It's really fun! You can choose to be a doctor, a lawyer, a teacher, or anything else you want.\nUser 1: That sounds cool! I'll have to check it out.\nUser 2: You should! It's a lot of fun.\nUser 1: Thanks for the recommendation!\nUser 2: No problem!",
    // "User 1: My mom was a nurse too! What did she like about her job?\nUser 2: She loved helping people. She also loved the variety of work that nurses do.\nUser 1: That's great! I have a cat. What kind of animals do you like?\nUser 2: I love all animals, but I especially love dogs. I had a dog named Max when I was a kid.\nUser 1: I love dogs too! I used to have a dog named Shadow.\nUser 2: What kind of dog was Shadow?\nUser 1: He was a black lab.\nUser 2: Oh, I love black labs! They're so cute.\nUser 1: They are! Shadow was the best dog ever. I miss him a lot.\nUser 2: I'm sure he was. I'm sorry to hear that you lost him.\nUser 1: Thank you.\nUser 2: So, what do you like to do for fun?\nUser 1: I like to sew, read, and go to the movies.\nUser 2: I like to cook, garden, and go hiking.\nUser 1: I love to cook too! We should exchange recipes sometime.\nUser 2: That would be fun!\nUser 1: What's your favorite thing to cook?\nUser 2: I love to make lasagna.\nUser 1: Lasagna is my favorite!\nUser 2: We're going to have to make lasagna together sometime.\nUser 1: That would be great!",
};

const char *droid_task[] = {
    "records:\n- Choice: 5\nInput: 'null'\nState: '<input id=0>Search</input>\n\n<button id=1 text=''Settings''></button>\n\n<button id=2 text=''About''></button>\n\n<button id=3 text=''More options''></button>\n\n<button id=4>Bob<br>Hey<br>30.06</button>\n\n<button id=5>Alice<br>Hello<br>30.06</button>\n\n<button id=6>go back</button>'\nnew_state_str: fc3f74bf7d6eb2bf6fa69fef7f791f434acac4448c9f2aac78a0820fd4b348f9\nstate_str:\n- ebe04be39b2a5afdab7abd3dfa557ba5be810fc9e9fc0bd29f0c6585511139db\n- Choice: -1\nInput: 'null'\nState: \"<p id=0>Alice</p>\\n <button id=1 text='Delete'></button>\\n <button id=2\\\n\\ text='Dial number'></button>\\n <button id=3 text='Add Person'></button>\\n <button\\\n\\ id=4 text='More options'></button>\\n <button id=5>30.06, 12:18 PM</button>\\n\\\n\\ <button id=6>Hello</button>\\n <button id=7 text='Attachment'></button>\\n <input\\\n\\ id=8 >Type a message\\u2026</input>\\n <button id=9 text='OK'>SMS</button>\\n <button\\\n\\ id=10>go back</button>\"\nnew_state_str: f6018d3c10b04f67973dad0e561b1ada96a741e8b69e7bc99cf265c7529802d0\nstate_str: aeda9619bbd29160179a173542c593dc23dc47b56b6b46f2a5fb631b7fbc9bfa\nstep_num: 4\ntask_name: find messages with Alice\n",
    "records:\n- Choice: 0\nInput: Clock\nState: \"<input id=0 >Search</input>\\n <button id=1 text='Sort by'></button>\\n <button\\\n\\ id=2 text='Toggle app name visibility'></button>\\n <button id=3 text='More options'></button>\\n\\\n\\ <button id=4>Calendar</button>\\n <button id=5>Camera</button>\\n <button id=6>Clock</button>\\n\\\n\\ <button id=7>go back</button>\"\nnew_state_str: 3f9adbf098ff69152b801042222f25ddd612674644e5d20b4c3617b4c9e33b3f\nstate_str: ca84887fa50fbe8cb13d9c9fbcd7f1e568612b5e7448bbc9ac0425944993c94c\n- Choice: 4\nInput: 'null'\nState: \"<input id=0 >Clock</input>\\n <button id=1 text='Sort by'></button>\\n <button\\\n\\ id=2 text='Toggle app name visibility'></button>\\n <button id=3 text='More options'></button>\\n\\\n\\ <button id=4>Clock</button>\\n <button id=5>go back</button>\"\nnew_state_str: ef4cb2b7f5b1e85ac7426d6f120709e1a0e67a7c8b0bc90bdd9a90ed3beb1f60\nstate_str: acf566b0abeeca2407aab2ef15cb101bf78a2b81b0e1ddcce4049f5950783e11\n- Choice: -1\nInput: 'null'\nState: '<button id=0 text=''More apps from us''></button>\n\n<button id=1 text=''Settings''></button>\n\n<button id=2 text=''About''></button>\n\n<button id=3>05:00</button>\n\n<button id=4 text=''New Timer''></button>\n\n<button id=5>Clock</button>\n\n<button id=6>Alarm</button>\n\n<button id=7>Stopwatch</button>\n\n<p id=8>Timer</p>\n\n<button id=9>go back</button>'\nnew_state_str: fb78df2efd2d0dae6fe2d27fc76d3c48cf2944110abca15ccb49178d2e962f30\nstate_str:\n- 8fe6b1742232194edf2780f0ef572fa72b1403eda181620060b0c46061a21ef7\nstep_num: 4\ntask_name: search 'Clock' app and open it\n",
    "records:\n- Choice: 1\nInput: 'null'\nState: \"<input id=0 >Search</input>\\n <button id=1 text='Sort by'></button>\\n <button\\\n\\ id=2 text='Toggle app name visibility'></button>\\n <button id=3 text='More options'></button>\\n\\\n\\ <button id=4>Calendar</button>\\n <button id=5>Camera</button>\\n <button id=6>Clock</button>\\n\\\n\\ <button id=7>go back</button>\"\nnew_state_str: 3f9adbf098ff69152b801042222f25ddd612674644e5d20b4c3617b4c9e33b3f\nstate_str: ca84887fa50fbe8cb13d9c9fbcd7f1e568612b5e7448bbc9ac0425944993c94c\n- Choice: 4\nInput: 'null'\nState: \"<p id=0>Sort by</p>\\n <checkbox id=1 checked=True>Title</checkbox>\\n <checkbox\\\n\\ id=2 checked=False>Custom</checkbox>\\n <checkbox id=3 checked=True>Ascending</checkbox>\\n\\\n\\ <checkbox id=4 checked=False>Descending</checkbox>\\n <button id=5>Cancel</button>\\n\\\n\\ <button id=6>OK</button>\\n <button id=7>go back</button>\"\nnew_state_str: 7bd9e2ded8ae3ec34dffeefb946dad82da73e3a8c7d983d8eed62b7ba2093beb\nstate_str: 70bfe9da5a48f69d777a60eb66784f548c873cca4a9e34134ee7c9a6262d1e89\n- Choice: 6\nInput: 'null'\nState: \"<p id=0>Sort by</p>\\n <checkbox id=1 checked=True>Title</checkbox>\\n <checkbox\\\n\\ id=2 checked=False>Custom</checkbox>\\n <checkbox id=3 checked=False>Ascending</checkbox>\\n\\\n\\ <checkbox id=4 checked=True>Descending</checkbox>\\n <button id=5>Cancel</button>\\n\\\n\\ <button id=6>OK</button>\\n <button id=7>go back</button>\"\nnew_state_str: 1d1c21762d4842a6f0e455ce1321480e39d0413cdda62f58f51f742bedf2f194\nstate_str: 313df1fcaaa12eca00c9156d1119ebc2a246c689938a683bb0374ec09bb6922f\n- Choice: -1\nInput: 'null'\nState: \"<input id=0 >Search</input>\\n <button id=1 text='Sort by'></button>\\n <button\\\n\\ id=2 text='Toggle app name visibility'></button>\\n <button id=3 text='More options'></button>\\n\\\n\\ <button id=4>Clock</button>\\n <button id=5>Camera</button>\\n <button id=6>Calendar</button>\\n\\\n\\ <button id=7>go back</button>\"\nnew_state_str: b5327a6ca3fab812a0c123130a8dbab27a1168406851f9e1afc30a08ce631515\nstate_str: 92555ffd38a0f4bf048b06d1b0e8d426ba7418fee5fe79b6bcccf8fca09e9c48\nstep_num: 4\ntask_name: Sort apps by title in descending order\n",
    "records:\n- Choice: 6\nInput: 'null'\nState: \"<button id=0 text='Toggle the timer mode'></button>\\n <button id=1 text='Resolution'></button>\\n\\\n\\ <button id=2 text='Settings'></button>\\n <button id=3 text='Video'>Video</button>\\n\\\n\\ <p id=4 text='Photo'></p>\\n <p id=5>Photo</p>\\n <button id=6 text='Toggle front/back\\\n\\ camera'></button>\\n <button id=7 text='Shutter'></button>\\n <button id=8 text='View\\\n\\ last captured media'></button>\\n <button id=9>go back</button>\"\nnew_state_str: 863cb466e6a7b76e3579075348e7500c30d85b1deecbd22fd9bad4fcb02cdbfb\nstate_str: 51df7fe2285dc69df6c3991ff2a7245a83238ce0a9f5457cf10100669270ca7a\n- Choice: 8\nInput: 'null'\nState: \"<button id=0 text='Toggle the timer mode'></button>\\n <button id=1 text='Toggle\\\n\\ the flashlight mode'></button>\\n <button id=2 text='Resolution'></button>\\n\\\n\\ <button id=3 text='Settings'></button>\\n <button id=4 text='Video'>Video</button>\\n\\\n\\ <p id=5 text='Photo'></p>\\n <p id=6>Photo</p>\\n <button id=7 text='Toggle front/back\\\n\\ camera'></button>\\n <button id=8 text='Shutter'></button>\\n <button id=9 text='View\\\n\\ last captured media'></button>\\n <button id=10>go back</button>\"\nnew_state_str: 6b487e7860b4bcc2b78505508b20891f3434e16e473a55e614b91c875249e809\nstate_str: 89ec00c5a4b58fb355df5679f98f56cc3100526ff698b73e551c5d21763061e5\n- Choice: -1\nInput: 'null'\nState: \"<button id=0 text='Toggle the timer mode'></button>\\n <button id=1 text='Toggle\\\n\\ the flashlight mode'></button>\\n <button id=2 text='Resolution'></button>\\n\\\n\\ <button id=3 text='Settings'></button>\\n <button id=4 text='Video'>Video</button>\\n\\\n\\ <p id=5 text='Photo'></p>\\n <p id=6>Photo</p>\\n <button id=7 text='Toggle front/back\\\n\\ camera'></button>\\n <button id=8 text='Shutter'></button>\\n <button id=9 text='View\\\n\\ last captured media'></button>\\n <button id=10>go back</button>\"\nnew_state_str: 6b487e7860b4bcc2b78505508b20891f3434e16e473a55e614b91c875249e809\nstate_str: 89ec00c5a4b58fb355df5679f98f56cc3100526ff698b73e551c5d21763061e5\nstep_num: 3\ntask_name: take a selfie\n",
    "records:\n- Choice: 5\nInput: 'null'\nState: '<button id=0 text=''Enable private browsing''></button>\n\n<p id=1 text=''Search or enter address''></p>\n\n<button id=2 text=''Google''></button>\n\n<p id=3>Search or enter address</p>\n\n<button id=4 text=''0 open tabs. Tap to switch tabs.<br>The tab counter toolbar\nbutton.''>0</button>\n\n<button id=5 text=''Menu<br>Highlighted''></button>\n\n<button id=6>go back</button>'\nnew_state_str: 8296582e793e9f504816578383d9d54e13b7fac650c9089e45589743940ebf27\nstate_str:\n- a3cb8fb13bd7c9669e226f206b603089b175e197100a1cf07bb1db68002a0d16\n- a3cb8fb13bd7c9669e226f206b603089b175e197100a1cf07bb1db68002a0d16\n- Choice: 2\nInput: 'null'\nState: \"<button id=0>Bookmarks</button>\\n <button id=1>History</button>\\n <button\\\n\\ id=2>Downloads</button>\\n <button id=3>Add-ons</button>\\n <button id=4>Sync\\\n\\ and save data</button>\\n <button id=5>Manage Account and Devices</button>\\n\\\n\\ <checkbox id=6 checked=False>Desktop site</checkbox>\\n <button id=7 text='Highlighted<br>Highlighted,\\\n\\ What\\u2019s new'>What\\u2019s new</button>\\n <button id=8>Help</button>\\n <button\\\n\\ id=9>Customize homepage</button>\\n <button id=10>Settings</button>\\n <button\\\n\\ id=11>go back</button>\"\nnew_state_str: 39180632bed6820472c6b99209b673b6fc599e7f661164ee753d0964854383bf\nstate_str: 9d1efab623f99dfee6d98a5ef2d601b53e602d98cfdab3f2d3384ff8ca11dcff\n- Choice: -1\nInput: 'null'\nState: \"<button id=0 text='Navigate up'></button>\\n <p id=1>Downloads</p>\\n <button\\\n\\ id=2 text='Close'></button>\\n <p id=3>No downloaded files</p>\\n <button id=4>go\\\n\\ back</button>\"\nnew_state_str: 0c93dc1459af0daa90156ba17264816e789a9755e12d45240bef26020ef0fa56\nstate_str: e2d80544e1f15a9f3a517b557088bf7dc37febe7953831462fa3e3e574ac0afe\nstep_num: 3\ntask_name: view files dowloaded through firefox\n",
    // "records:\n- Choice: 5\nInput: 'null'\nState: '<input id=0>Search folders</input>\n\n<button id=1 text=''Open camera''></button>\n\n<button id=2 text=''Show all folders content''></button>\n\n<button id=3 text=''More options''></button>\n\n<button id=4>Internal<br>2</button>\n\n<button id=5>DCIM<br>6</button>\n\n<button id=6>go back</button>'\nnew_state_str: a54edb4560e79ad038ec5120a09ec838b034edd416abba65385969613b18a95b\nstate_str:\n- 073664fc497d71c2d6fbd0680dbdd396ec5cf3939cd9a21af17a8da3b79c7f67\n- Choice: -1\nInput: 'null'\nState: '<input id=0>Search in DCIM</input>\n\n<button id=1 text=''Toggle filename visibility''></button>\n\n<button id=2 text=''Sort by''></button>\n\n<button id=3 text=''More options''></button>\n\n<button id=4>go back</button>'\nnew_state_str: c4867c874a83f1245254d117e2f3761349f3beeeedd4fb03c66f737ccafa242a\nstate_str:\n- 47203338b2fa5e21d22f1397ac0d72d9e74179220d59ef833c7ad7889aaa726d\nstep_num: 2\ntask_name: Open DCIM folder in gallery app\n",
    // "records:\n- Choice: 5\nInput: 'null'\nState: '<input id=0>Search</input>\n\n<button id=1 text=''Settings''></button>\n\n<button id=2 text=''About''></button>\n\n<button id=3 text=''More options''></button>\n\n<button id=4>Bob<br>Hey<br>30.06</button>\n\n<button id=5>Alice<br>Hello<br>30.06</button>\n\n<button id=6>go back</button>'\nnew_state_str: fc3f74bf7d6eb2bf6fa69fef7f791f434acac4448c9f2aac78a0820fd4b348f9\nstate_str:\n- ebe04be39b2a5afdab7abd3dfa557ba5be810fc9e9fc0bd29f0c6585511139db\n- Choice: -1\nInput: 'null'\nState: \"<p id=0>Alice</p>\\n <button id=1 text='Delete'></button>\\n <button id=2\\\n\\ text='Dial number'></button>\\n <button id=3 text='Add Person'></button>\\n <button\\\n\\ id=4 text='More options'></button>\\n <button id=5>30.06, 12:18 PM</button>\\n\\\n\\ <button id=6>Hello</button>\\n <button id=7 text='Attachment'></button>\\n <input\\\n\\ id=8 >Type a message\\u2026</input>\\n <button id=9 text='OK'>SMS</button>\\n <button\\\n\\ id=10>go back</button>\"\nnew_state_str: f6018d3c10b04f67973dad0e561b1ada96a741e8b69e7bc99cf265c7529802d0\nstate_str: aeda9619bbd29160179a173542c593dc23dc47b56b6b46f2a5fb631b7fbc9bfa\nstep_num: 4\ntask_name: find messages with Alice\n",
    // "records:\n- Choice: 3\nInput: 'null'\nState: '<button id=0 text=''Search''></button>\n\n<button id=1 text=''Open note''></button>\n\n<button id=2 text=''Create a new note''></button>\n\n<button id=3 text=''More options''></button>\n\n<p id=4>Diary</p>\n\n<button id=5>General note</button>\n\n<input id=6>2023.6.10 Sunny</input>\n\n<button id=7>go back</button>'\nnew_state_str: 8dea304280d163f83acfaf5361f23bd69f0ee73b1e8d49e8b209e51e2f701476\nstate_str:\n- 7dd599d9b413a75be1c426d9601dea75356fae2f5064cf4377daf3cb209125a6\n- Choice: 9\nInput: 'null'\nState: \"<button id=0>Rename note</button>\\n <button id=1>Share</button>\\n <button\\\n\\ id=2>Create shortcut</button>\\n <button id=3>Lock note</button>\\n <button id=4>Open\\\n\\ file</button>\\n <button id=5>Import notes</button>\\n <button id=6>Export as\\\n\\ file</button>\\n <button id=7>Export all notes</button>\\n <button id=8>Print</button>\\n\\\n\\ <button id=9>Delete note</button>\\n <button id=10>Settings</button>\\n <button\\\n\\ id=11>About</button>\\n <button id=12>go back</button>\"\nnew_state_str: d30d629d793bfe53cdcfb3f2984315c78089bafee2ba0f45ef8bc2c0dcf7909a\nstate_str: 4e5778fa48b7653d78c78d9fc93185b3bf480d25f784957a6e6ce8b90ed6573d\n- Choice: 2\nInput: 'null'\nState: \"<p id=0>Are you sure you want to delete note \\\"Diary\\\"?</p>\\n <button id=1>Cancel</button>\\n\\\n\\ <button id=2>Delete</button>\\n <button id=3>go back</button>\"\nnew_state_str: fc6060f782647d6f5d40da6d29994f951b643b36ad707888bc79a8f35db6323c\nstate_str: 1690bc0d85368814cfabdc33e9b0185f2899e20e221ab6b9171ca0d85650435f\n- Choice: -1\nInput: 'null'\nState: \"<button id=0 text='Search'></button>\\n <button id=1 text='Create a new note'></button>\\n\\\n\\ <button id=2 text='More options'></button>\\n <input id=3 >Insert text here</input>\\n\\\n\\ <button id=4>go back</button>\"\nnew_state_str: 7b0e78b3a6b4f26152d48b0628eb7fd72fd86ee1843c042516df15de34a3db86\nstate_str: 34cc245f55ca3016faf4a558b3170984f5cab37677d453dd52802f8a3cd1b3be\nstep_num: 4\ntask_name: Delete the note 'Diary'\n",
    // "records:\n- Choice: 3\nInput: 'null'\nState: \"<input id=0 >Search</input>\\n <button id=1 text='Sort by'></button>\\n <button\\\n\\ id=2 text='Toggle app name visibility'></button>\\n <button id=3 text='More options'></button>\\n\\\n\\ <button id=4>Calendar</button>\\n <button id=5>Camera</button>\\n <button id=6>Clock</button>\\n\\\n\\ <button id=7>go back</button>\"\nnew_state_str: 3f9adbf098ff69152b801042222f25ddd612674644e5d20b4c3617b4c9e33b3f\nstate_str: ca84887fa50fbe8cb13d9c9fbcd7f1e568612b5e7448bbc9ac0425944993c94c\n- Choice: 1\nInput: 'null'\nState: \"<button id=0>Column count</button>\\n <button id=1>Settings</button>\\n <button\\\n\\ id=2>About</button>\\n <button id=3>go back</button>\"\nnew_state_str: ca19303f69337e3b0304675f3593b6929a8f5e20f11c851c7caa0c01e2149f4c\nstate_str: 4aa00a0b2088cff9cf01711cc84ced31f46a408b7dd1e4bc992e0e14003ec750\n- Choice: 4\nInput: 'null'\nState: \"<p id=0>Settings</p>\\n <p id=1>COLOR CUSTOMIZATION</p>\\n <button id=2>Customize\\\n\\ colors</button>\\n <p id=3>GENERAL</p>\\n <checkbox id=4 checked=True>Close this\\\n\\ app at launching a different one</checkbox>\\n <button id=5>go back</button>\"\nnew_state_str: c1b06bce7e23dd13a3288977fbdd55b3d1962a9bb93432a712321fd371f52bb7\nstate_str: ad13fc2aeb3d2bc44cc46c16bec9cd521ce8c4a2109a8c41b6e3186a67e681bd\n- Choice: -1\nInput: 'null'\nState: \"<p id=0>Settings</p>\\n <p id=1>COLOR CUSTOMIZATION</p>\\n <button id=2>Customize\\\n\\ colors</button>\\n <p id=3>GENERAL</p>\\n <checkbox id=4 checked=False>Close this\\\n\\ app at launching a different one</checkbox>\\n <button id=5>go back</button>\"\nnew_state_str: 6948b52ae5a0012089344c00a0b0521ab8778828b8b4ac3c8552aaa62deb1d77\nstate_str: 856403056505397d1789eb60a5a2a01188ea16b63867d9414e52471da38bd03b\nstep_num: 4\ntask_name: Disable closing this app when launching a different one\n",
    // "records:\n- Choice: 8\nInput: 'null'\nState: \"<button id=0 text='Toggle the timer mode'></button>\\n <button id=1 text='Resolution'></button>\\n\\\n\\ <button id=2 text='Settings'></button>\\n <button id=3 text='Video'>Video</button>\\n\\\n\\ <p id=4 text='Photo'></p>\\n <p id=5>Photo</p>\\n <button id=6 text='Toggle front/back\\\n\\ camera'></button>\\n <button id=7 text='Shutter'></button>\\n <button id=8 text='View\\\n\\ last captured media'></button>\\n <button id=9>go back</button>\"\nnew_state_str: 863cb466e6a7b76e3579075348e7500c30d85b1deecbd22fd9bad4fcb02cdbfb\nstate_str: 51df7fe2285dc69df6c3991ff2a7245a83238ce0a9f5457cf10100669270ca7a\n- Choice: -1\nInput: 'null'\nState: '<p id=0>pic3.jpg</p>\n\n<button id=1 text=''Rotate''></button>\n\n<button id=2 text=''Properties''></button>\n\n<button id=3 text=''More options''></button>\n\n<button id=4 text=''Toggle favorite''></button>\n\n<button id=5 text=''Edit''></button>\n\n<button id=6 text=''Share''></button>\n\n<button id=7 text=''Delete''></button>\n\n<button id=8>go back</button>'\nnew_state_str: f0c7d4f5cd80d174486b8bf80b9d8fdf9194087dba2597e5a56418705f1d28e1\nstate_str:\n- 1d423634a7686aeb54dcd3e9084178d9176e4c73e2d71ccafb069c34f1aa46b0\nstep_num: 2\ntask_name: open the last picture I took\n",
};

void parse_prompt(gpt_params & params, const std::string & value) {
    // params.prompt = value;
    std::cout << "prompt " << value << std::endl;
    size_t pos = value.find('#');
    if (pos != std::string::npos) {
        std::string model = value.substr(0, pos);
        int len = std::stoi(value.substr(pos + 1));
        if (model == "tinyllama") {
            params.io_model_path = "/data/ssd/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
        } else if (model == "gemma") {
            params.io_model_path = "/data/ssd/gemma-2-2b-it-Q8_0.gguf";
        } else if (model == "qwen") {
            params.io_model_path = "/data/ssd/qwen2.5-3b-instruct-q8_0.gguf";
        } else if (model == "phi") {
            params.io_model_path = "/data/ssd/Phi-3-mini-4k-instruct.Q8_0.gguf";
        } else if (model == "llama") {
            params.io_model_path = "/data/ssd/Meta-Llama-3-8B-Instruct.Q8_0.gguf";
        } else {
            GGML_ABORT("invalid prompt %s\n", value);
        }
        
        GGML_ASSERT(len >= 0);
        if (len < sizeof(ultra_chat_prompts) / sizeof(*ultra_chat_prompts)) {
            params.prompt = std::string(ultra_chat_prompts[len]);
        } else {
            len -= sizeof(ultra_chat_prompts) / sizeof(*ultra_chat_prompts);
            if (len < sizeof(persona_c) / sizeof(*persona_c)) {
                char tmp[2000];
                sprintf(
                    tmp,
                    "Here, we list the profiles of two users, user 1 and user 2, followed by an interesting and natural conversation between user 1 and user 2, which implicitly reflects their user profiles.\nUser 1 Profile: %s\nUser 2 Profile: %s\nConversation: %s\nGive me more examples like this. The conversation must be more than 5 turns and less than 8 turns. The conversation must be natural, and not direct copies of their profiles.\n",
                    persona_u1[len], persona_u2[len], persona_c[len]
                );
                params.prompt = std::string(tmp);
            } else {
                len -= sizeof(persona_c) / sizeof(*persona_c);
                if (len < sizeof(droid_task) / sizeof(*droid_task)) {
                    params.prompt = std::string(droid_task[len]);
                } else {
                    goto out;
                }
            }
        }
        return;
out:
        len += sizeof(ultra_chat_prompts) / sizeof(*ultra_chat_prompts);
        len += sizeof(persona_c) / sizeof(*persona_c);


        if (model == "tinyllama") {
            params.io_model_path = "/data/ssd/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
            switch (len) {
            case 32:
                params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should";
                break;
            case 64:
                params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should";
                break;
            case 128:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power to uplift spirits.\n";
                break;
            case 256:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power to uplift spirits.\n Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power to uplift spirits.\n";
                break;
            case 512:
                params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box?\n";
                break;
            default:
                GGML_ABORT("invalid len %d", len);
            }
        } else if (model == "gemma") {
            params.io_model_path = "/data/ssd/gemma-2-2b-it-Q8_0.gguf";
            switch (len) {
            case 32:
                params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email";
                break;
            case 128:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits.\nWrite a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits.\n";
                break;
            case 512:
                params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove ?\n";
                break;
            default:
                GGML_ABORT("invalid len %d", len);
            }
        } else if (model == "qwen") {
            params.io_model_path = "/data/ssd/qwen2.5-3b-instruct-q8_0.gguf";
            switch (len) {
            case 1:
                params.prompt = "I I";
                break;
            case 32:
                params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how";
                break;
            case 64:
                params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how";
                break;
            case 128:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and.";
                break;
            case 192:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how";
                break;
            case 256:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and." "\n" "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and.";
                break;
            case 320:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and." "\n" "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how";
                break;
            case 384:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and." "\n" "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and.\n" "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and.";
                break;
            case 448:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and." "\n" "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and.\n" "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how";
                break;
            case 512:
                params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box \n";
                break;
            default:
                GGML_ABORT("invalid len %d", len);
            }
        } else if (model == "phi") {
            params.io_model_path = "/data/ssd/Phi-3-mini-4k-instruct.Q8_0.gguf";
            switch (len) {
            case 32:
                params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should ";
                break;
            case 128:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits.\nWrite a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language and at least one metaphor or simile.\n";
                break;
            case 512:
                params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How?\n";
                break;
            default:
                GGML_ABORT("invalid len %d", len);
            }
        } else if (model == "llama") {
            params.io_model_path = "/data/ssd/Meta-Llama-3-8B-Instruct.Q8_0.gguf";
            switch (len) {
            case 1:
                params.prompt = "I";
                break;
            case 32:
                params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 64:
                params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 96:
                params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 128:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect.";
                break;
            case 160:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 192:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 224:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 256:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect.";
                break;
            case 288:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 320:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 352:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 384:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect.";
                break;
            case 416:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 448:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 480:
                params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect. If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
                break;
            case 512:
                params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box. How many plates did \n";
                break;
            default:
                GGML_ABORT("invalid len %d", len);
            }
        } else {
            GGML_ABORT("invalid prompt %s\n", value);
        }
    } else {
        GGML_ABORT("invalid prompt %s\n", value);
    }
    {
        // tinyllama-1.1b-chat-v1.0.Q8_0
        // params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should";
        // params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power to uplift spirits.\n";
        // params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box?\n";
    }

    {
        // gemma-2-2b-it-Q8_0
        // params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email";
        // params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits.\nWrite a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits.\n";
        // params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove ?\n";

    }

    {
        // qwen2.5-3b-instruct-q8_0
        // 32
        // params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on how";
        // 128
        // params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and.";
        // 256
        // params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and." "\n" "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and.";
        // 384
        // params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and." "\n" "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and.\n" "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and.";
        // 512
        // params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box \n";
    }

    {
        // Phi-3-mini-4k-instruct.Q8_0
        // params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should ";
        // params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits.\nWrite a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language and at least one metaphor or simile.\n";
        // params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How?\n";
    }

    {
        // Meta-Llama-3-8B-Instruct.Q8_0
        // 32
        // params.prompt = "If you are submitting a revised paper because you received a revise-and-resubmit decision from the previous deadline, you should already have received instructions by email on";
        // 128
        // params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect.";
        // 256
        // params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect.";
        // 384
        // params.prompt = "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect." "Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect people. Write a 10-line free verse poem about the contagious joys of laughter with a focus on the physical and emotional benefits it provides. Use descriptive language to evoke a vivid image of a joyful moment shared among friends, and include at least one metaphor or simile to express the power of laughter to uplift spirits and connect.";
        // 512
        // params.prompt = "Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? Hasan is packing up his apartment because he’s moving across the country for a new job. He needs to ship several boxes to his new home. The movers have asked that Hasan avoid putting more than a certain weight in pounds in any cardboard box. The moving company has helpfully provided Hasan with a digital scale that will alert him if a package is too heavy. Hasan is in the kitchen, and he fills a cardboard box with 38 dinner plates. When he checks the box, the scale reports his box is too heavy. Hasan knows each of his plates weighs 10 ounces. He removes a single plate from the box and checks the movers’ scale again. The scale reports his box is still too heavy. Hasan repeats the process again and again. When he has removed enough plates, the movers’ scale shows the box is now an acceptable weight for shipping. Hasan deduces that each shipping box can hold 20 pounds before the scale says the box is too heavy. How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box? How many plates did Hasan need to remove from the shipping box. How many plates did \n";
    }

}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params, llama_example ex, void(*print_usage)(int, char **)) {

    std::cout << "In function gpt_params_parse" << std::endl;

    auto ctx_arg = gpt_params_parser_init(params, ex, print_usage);
    const gpt_params params_org = ctx_arg.params; // the example can modify the default params

    std::cout << "After gpt_params_parser_init" << std::endl;

    try {
        if (!gpt_params_parse_ex(argc, argv, ctx_arg)) {
            ctx_arg.params = params_org;
            return false;
        }
        std::cout << "After gpt_params_parse_ex" << std::endl;

        if (ctx_arg.params.usage) {
            gpt_params_print_usage(ctx_arg);
            if (ctx_arg.print_usage) {
                ctx_arg.print_usage(argc, argv);
            }
            exit(0);
        }
        std::cout << "After gpt_params_print_usage" << std::endl;
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        ctx_arg.params = params_org;
        return false;
    }

    return true;
}

gpt_params_context gpt_params_parser_init(gpt_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    gpt_params_context ctx_arg(params);
    ctx_arg.print_usage = print_usage;
    ctx_arg.ex          = ex;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto & sampler : params.sparams.samplers) {
        sampler_type_chars += gpt_sampler_type_to_chr(sampler);
        sampler_type_names += gpt_sampler_type_to_str(sampler) + ";";
    }
    sampler_type_names.pop_back();

    std::cout << "After gpt_params_context" << std::endl;
    /**
     * filter options by example
     * rules:
     * - all examples inherit options from LLAMA_EXAMPLE_COMMON
     * - if LLAMA_EXAMPLE_* is set (other than COMMON), we only show the option in the corresponding example
     * - if both {LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_*,} are set, we will prioritize the LLAMA_EXAMPLE_* matching current example
     */
    auto add_opt = [&](llama_arg arg) {
        if (arg.in_example(ex) || arg.in_example(LLAMA_EXAMPLE_COMMON)) {
            ctx_arg.options.push_back(std::move(arg));
        }
    };


    add_opt(llama_arg(
        {"-h", "--help", "--usage"},
        "print usage and exit",
        [](gpt_params & params) {
            params.usage = true;
        }
    ));
    add_opt(llama_arg(
        {"--version"},
        "show version and build info",
        [](gpt_params &) {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        }
    ));
    add_opt(llama_arg(
        {"--verbose-prompt"},
        format("print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false"),
        [](gpt_params & params) {
            params.verbose_prompt = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--no-display-prompt"},
        format("don't print prompt at generation (default: %s)", !params.display_prompt ? "true" : "false"),
        [](gpt_params & params) {
            params.display_prompt = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-co", "--color"},
        format("colorise output to distinguish prompt and user input from generations (default: %s)", params.use_color ? "true" : "false"),
        [](gpt_params & params) {
            params.use_color = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL, LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP}));
    add_opt(llama_arg(
        {"-t", "--threads"}, "N",
        format("number of threads to use during generation (default: %d)", params.cpuparams.n_threads),
        [](gpt_params & params, int value) {
            params.cpuparams.n_threads = value;
            if (params.cpuparams.n_threads <= 0) {
                params.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_env("LLAMA_ARG_THREADS"));
    add_opt(llama_arg(
        {"-tb", "--threads-batch"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads)",
        [](gpt_params & params, int value) {
            params.cpuparams_batch.n_threads = value;
            if (params.cpuparams_batch.n_threads <= 0) {
                params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ));
    add_opt(llama_arg(
        {"-td", "--threads-draft"}, "N",
        "number of threads to use during generation (default: same as --threads)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams.n_threads = value;
            if (params.draft_cpuparams.n_threads <= 0) {
                params.draft_cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-tbd", "--threads-batch-draft"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads-draft)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams_batch.n_threads = value;
            if (params.draft_cpuparams_batch.n_threads <= 0) {
                params.draft_cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-C", "--cpu-mask"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
        [](gpt_params & params, const std::string & mask) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(llama_arg(
        {"-Cr", "--cpu-range"}, "lo-hi",
        "range of CPUs for affinity. Complements --cpu-mask",
        [](gpt_params & params, const std::string & range) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(llama_arg(
        {"--cpu-strict"}, "<0|1>",
        format("use strict CPU placement (default: %u)\n", (unsigned) params.cpuparams.strict_cpu),
        [](gpt_params & params, const std::string & value) {
            params.cpuparams.strict_cpu = std::stoul(value);
        }
    ));
    add_opt(llama_arg(
        {"--prio"}, "N",
        format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams.priority),
        [](gpt_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(llama_arg(
        {"--poll"}, "<0...100>",
        format("use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) params.cpuparams.poll),
        [](gpt_params & params, const std::string & value) {
            params.cpuparams.poll = std::stoul(value);
        }
    ));
    add_opt(llama_arg(
        {"-Cb", "--cpu-mask-batch"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
        [](gpt_params & params, const std::string & mask) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(llama_arg(
        {"-Crb", "--cpu-range-batch"}, "lo-hi",
        "ranges of CPUs for affinity. Complements --cpu-mask-batch",
        [](gpt_params & params, const std::string & range) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(llama_arg(
        {"--cpu-strict-batch"}, "<0|1>",
        "use strict CPU placement (default: same as --cpu-strict)",
        [](gpt_params & params, int value) {
            params.cpuparams_batch.strict_cpu = value;
        }
    ));
    add_opt(llama_arg(
        {"--prio-batch"}, "N",
        format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams_batch.priority),
        [](gpt_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(llama_arg(
        {"--poll-batch"}, "<0|1>",
        "use polling to wait for work (default: same as --poll)",
        [](gpt_params & params, int value) {
            params.cpuparams_batch.poll = value;
        }
    ));
    add_opt(llama_arg(
        {"-Cd", "--cpu-mask-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](gpt_params & params, const std::string & mask) {
            params.draft_cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.draft_cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-Crd", "--cpu-range-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
        [](gpt_params & params, const std::string & range) {
            params.draft_cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.draft_cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--cpu-strict-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: same as --cpu-strict)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams.strict_cpu = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--prio-draft"}, "N",
        format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.draft_cpuparams.priority),
        [](gpt_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.draft_cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--poll-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: same as --poll])",
        [](gpt_params & params, int value) {
            params.draft_cpuparams.poll = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-Cbd", "--cpu-mask-batch-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](gpt_params & params, const std::string & mask) {
            params.draft_cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.draft_cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-Crbd", "--cpu-range-batch-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)",
        [](gpt_params & params, const std::string & range) {
            params.draft_cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.draft_cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--cpu-strict-batch-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: --cpu-strict-draft)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams_batch.strict_cpu = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--prio-batch-draft"}, "N",
        format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.draft_cpuparams_batch.priority),
        [](gpt_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.draft_cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--poll-batch-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: --poll-draft)",
        [](gpt_params & params, int value) {
            params.draft_cpuparams_batch.poll = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"--draft"}, "N",
        format("number of tokens to draft for speculative decoding (default: %d)", params.n_draft),
        [](gpt_params & params, int value) {
            params.n_draft = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP}));
    add_opt(llama_arg(
        {"-ps", "--p-split"}, "N",
        format("speculative decoding split probability (default: %.1f)", (double)params.p_split),
        [](gpt_params & params, const std::string & value) {
            params.p_split = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-lcs", "--lookup-cache-static"}, "FNAME",
        "path to static lookup cache to use for lookup decoding (not updated by generation)",
        [](gpt_params & params, const std::string & value) {
            params.lookup_cache_static = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP}));
    add_opt(llama_arg(
        {"-lcd", "--lookup-cache-dynamic"}, "FNAME",
        "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
        [](gpt_params & params, const std::string & value) {
            params.lookup_cache_dynamic = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP}));
    add_opt(llama_arg(
        {"-c", "--ctx-size"}, "N",
        format("size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx),
        [](gpt_params & params, int value) {
            params.n_ctx = value;
        }
    ).set_env("LLAMA_ARG_CTX_SIZE"));
    add_opt(llama_arg(
        {"-n", "--predict", "--n-predict"}, "N",
        format("number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)", params.n_predict),
        [](gpt_params & params, int value) {
            params.n_predict = value;
        }
    ).set_env("LLAMA_ARG_N_PREDICT"));
    add_opt(llama_arg(
        {"-b", "--batch-size"}, "N",
        format("logical maximum batch size (default: %d)", params.n_batch),
        [](gpt_params & params, int value) {
            params.n_batch = value;
        }
    ).set_env("LLAMA_ARG_BATCH"));
    add_opt(llama_arg(
        {"-ub", "--ubatch-size"}, "N",
        format("physical maximum batch size (default: %d)", params.n_ubatch),
        [](gpt_params & params, int value) {
            params.n_ubatch = value;
        }
    ).set_env("LLAMA_ARG_UBATCH"));
    add_opt(llama_arg(
        {"--keep"}, "N",
        format("number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep),
        [](gpt_params & params, int value) {
            params.n_keep = value;
        }
    ));
    add_opt(llama_arg(
        {"--no-context-shift"},
        format("disables context shift on inifinite text generation (default: %s)", params.ctx_shift ? "disabled" : "enabled"),
        [](gpt_params & params) {
            params.ctx_shift = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--chunks"}, "N",
        format("max number of chunks to process (default: %d, -1 = all)", params.n_chunks),
        [](gpt_params & params, int value) {
            params.n_chunks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(llama_arg(
        {"-fa", "--flash-attn"},
        format("enable Flash Attention (default: %s)", params.flash_attn ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.flash_attn = true;
        }
    ).set_env("LLAMA_ARG_FLASH_ATTN"));
    add_opt(llama_arg(
        {"-p", "--prompt"}, "PROMPT",
        ex == LLAMA_EXAMPLE_MAIN
            ? "prompt to start generation with\nif -cnv is set, this will be used as system prompt"
            : "prompt to start generation with",
        [](gpt_params & params, const std::string & value) {
            parse_prompt(params, value);
        }
    ));
    add_opt(llama_arg(
        {"--cache"}, "CACHE",
        "cache proportion x/5 (default: 0)",
        [](gpt_params & params, const std::string & value) {
            int x = std::stoi(value);
            params.cache = x;
        }
    ));
    add_opt(llama_arg(
        {"--tee-shm-paddr"}, "PADDR",
        "tee-shm-paddr)",
        [](gpt_params & params, const std::string & value) {
            params.tee_shm_paddr = std::stoul(value);
        }
    ));
    add_opt(llama_arg(
        {"--no-perf"},
        format("disable internal libllama performance timings (default: %s)", params.no_perf ? "true" : "false"),
        [](gpt_params & params) {
            params.no_perf = true;
            params.sparams.no_perf = true;
        }
    ).set_env("LLAMA_ARG_NO_PERF"));
    add_opt(llama_arg(
        {"-f", "--file"}, "FNAME",
        "a file containing the prompt (default: none)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (!params.prompt.empty() && params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
    ));
    add_opt(llama_arg(
        {"--in-file"}, "FNAME",
        "an input file (repeat to specify multiple files)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            params.in_files.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"-bf", "--binary-file"}, "FNAME",
        "binary file containing the prompt (default: none)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::ostringstream ss;
            ss << file.rdbuf();
            params.prompt = ss.str();
            fprintf(stderr, "Read %zu bytes from binary file %s\n", params.prompt.size(), value.c_str());
        }
    ));
    add_opt(llama_arg(
        {"-e", "--escape"},
        format("process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", params.escape ? "true" : "false"),
        [](gpt_params & params) {
            params.escape = true;
        }
    ));
    add_opt(llama_arg(
        {"--no-escape"},
        "do not process escape sequences",
        [](gpt_params & params) {
            params.escape = false;
        }
    ));
    add_opt(llama_arg(
        {"-ptc", "--print-token-count"}, "N",
        format("print token count every N tokens (default: %d)", params.n_print),
        [](gpt_params & params, int value) {
            params.n_print = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--prompt-cache"}, "FNAME",
        "file to cache prompt state for faster startup (default: none)",
        [](gpt_params & params, const std::string & value) {
            params.path_prompt_cache = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--prompt-cache-all"},
        "if specified, saves user input and generations to cache as well\n",
        [](gpt_params & params) {
            params.prompt_cache_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--prompt-cache-ro"},
        "if specified, uses the prompt cache but does not update it",
        [](gpt_params & params) {
            params.prompt_cache_ro = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-r", "--reverse-prompt"}, "PROMPT",
        "halt generation at PROMPT, return control in interactive mode\n",
        [](gpt_params & params, const std::string & value) {
            params.antiprompt.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-sp", "--special"},
        format("special tokens output enabled (default: %s)", params.special ? "true" : "false"),
        [](gpt_params & params) {
            params.special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"-cnv", "--conversation"},
        format(
            "run in conversation mode:\n"
            "- does not print special tokens and suffix/prefix\n"
            "- interactive mode is also enabled\n"
            "(default: %s)",
            params.conversation ? "true" : "false"
        ),
        [](gpt_params & params) {
            params.conversation = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-i", "--interactive"},
        format("run in interactive mode (default: %s)", params.interactive ? "true" : "false"),
        [](gpt_params & params) {
            params.interactive = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-if", "--interactive-first"},
        format("run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false"),
        [](gpt_params & params) {
            params.interactive_first = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"-mli", "--multiline-input"},
        "allows you to write or paste multiple lines without ending each in '\\'",
        [](gpt_params & params) {
            params.multiline_input = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--in-prefix-bos"},
        "prefix BOS to user inputs, preceding the `--in-prefix` string",
        [](gpt_params & params) {
            params.input_prefix_bos = true;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--in-prefix"}, "STRING",
        "string to prefix user inputs with (default: empty)",
        [](gpt_params & params, const std::string & value) {
            params.input_prefix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"--in-suffix"}, "STRING",
        "string to suffix after user inputs with (default: empty)",
        [](gpt_params & params, const std::string & value) {
            params.input_suffix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"--no-warmup"},
        "skip warming up the model with an empty run",
        [](gpt_params & params) {
            params.warmup = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(llama_arg(
        {"--spm-infill"},
        format(
            "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)",
            params.spm_infill ? "enabled" : "disabled"
        ),
        [](gpt_params & params) {
            params.spm_infill = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"--samplers"}, "SAMPLERS",
        format("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)", sampler_type_names.c_str()),
        [](gpt_params & params, const std::string & value) {
            const auto sampler_names = string_split(value, ';');
            params.sparams.samplers = gpt_sampler_types_from_names(sampler_names, true);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"-s", "--seed"}, "SEED",
        format("RNG seed (default: %u, use random seed for %u)", params.sparams.seed, LLAMA_DEFAULT_SEED),
        [](gpt_params & params, const std::string & value) {
            params.sparams.seed = std::stoul(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--sampling-seq"}, "SEQUENCE",
        format("simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.sparams.samplers = gpt_sampler_types_from_chars(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--ignore-eos"},
        "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
        [](gpt_params & params) {
            params.sparams.ignore_eos = true;
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--penalize-nl"},
        format("penalize newline tokens (default: %s)", params.sparams.penalize_nl ? "true" : "false"),
        [](gpt_params & params) {
            params.sparams.penalize_nl = true;
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--temp"}, "N",
        format("temperature (default: %.1f)", (double)params.sparams.temp),
        [](gpt_params & params, const std::string & value) {
            params.sparams.temp = std::stof(value);
            params.sparams.temp = std::max(params.sparams.temp, 0.0f);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--top-k"}, "N",
        format("top-k sampling (default: %d, 0 = disabled)", params.sparams.top_k),
        [](gpt_params & params, int value) {
            params.sparams.top_k = value;
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--top-p"}, "N",
        format("top-p sampling (default: %.1f, 1.0 = disabled)", (double)params.sparams.top_p),
        [](gpt_params & params, const std::string & value) {
            params.sparams.top_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--min-p"}, "N",
        format("min-p sampling (default: %.1f, 0.0 = disabled)", (double)params.sparams.min_p),
        [](gpt_params & params, const std::string & value) {
            params.sparams.min_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--tfs"}, "N",
        format("tail free sampling, parameter z (default: %.1f, 1.0 = disabled)", (double)params.sparams.tfs_z),
        [](gpt_params & params, const std::string & value) {
            params.sparams.tfs_z = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--typical"}, "N",
        format("locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)", (double)params.sparams.typ_p),
        [](gpt_params & params, const std::string & value) {
            params.sparams.typ_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--repeat-last-n"}, "N",
        format("last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", params.sparams.penalty_last_n),
        [](gpt_params & params, int value) {
            params.sparams.penalty_last_n = value;
            params.sparams.n_prev = std::max(params.sparams.n_prev, params.sparams.penalty_last_n);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--repeat-penalty"}, "N",
        format("penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)", (double)params.sparams.penalty_repeat),
        [](gpt_params & params, const std::string & value) {
            params.sparams.penalty_repeat = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--presence-penalty"}, "N",
        format("repeat alpha presence penalty (default: %.1f, 0.0 = disabled)", (double)params.sparams.penalty_present),
        [](gpt_params & params, const std::string & value) {
            params.sparams.penalty_present = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--frequency-penalty"}, "N",
        format("repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)", (double)params.sparams.penalty_freq),
        [](gpt_params & params, const std::string & value) {
            params.sparams.penalty_freq = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--dynatemp-range"}, "N",
        format("dynamic temperature range (default: %.1f, 0.0 = disabled)", (double)params.sparams.dynatemp_range),
        [](gpt_params & params, const std::string & value) {
            params.sparams.dynatemp_range = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--dynatemp-exp"}, "N",
        format("dynamic temperature exponent (default: %.1f)", (double)params.sparams.dynatemp_exponent),
        [](gpt_params & params, const std::string & value) {
            params.sparams.dynatemp_exponent = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--mirostat"}, "N",
        format("use Mirostat sampling.\nTop K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n"
        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sparams.mirostat),
        [](gpt_params & params, int value) {
            params.sparams.mirostat = value;
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--mirostat-lr"}, "N",
        format("Mirostat learning rate, parameter eta (default: %.1f)", (double)params.sparams.mirostat_eta),
        [](gpt_params & params, const std::string & value) {
            params.sparams.mirostat_eta = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--mirostat-ent"}, "N",
        format("Mirostat target entropy, parameter tau (default: %.1f)", (double)params.sparams.mirostat_tau),
        [](gpt_params & params, const std::string & value) {
            params.sparams.mirostat_tau = std::stof(value);
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
        "modifies the likelihood of token appearing in the completion,\n"
        "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
        "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
        [](gpt_params & params, const std::string & value) {
            std::stringstream ss(value);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                    params.sparams.logit_bias.push_back({key, bias});
                } else {
                    throw std::invalid_argument("invalid input format");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument("invalid input format");
            }
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--grammar"}, "GRAMMAR",
        format("BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '%s')", params.sparams.grammar.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.sparams.grammar = value;
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--grammar-file"}, "FNAME",
        "file to read grammar from",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(params.sparams.grammar)
            );
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"-j", "--json-schema"}, "SCHEMA",
        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\nFor schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead",
        [](gpt_params & params, const std::string & value) {
            params.sparams.grammar = json_schema_to_grammar(json::parse(value));
        }
    ).set_sparam());
    add_opt(llama_arg(
        {"--pooling"}, "{none,mean,cls,last}",
        "pooling type for embeddings, use model default if unspecified",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
            else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
            else if (value == "cls") { params.pooling_type = LLAMA_POOLING_TYPE_CLS; }
            else if (value == "last") { params.pooling_type = LLAMA_POOLING_TYPE_LAST; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--attention"}, "{causal,non,causal}",
        "attention type for embeddings, use model default if unspecified",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "causal") { params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; }
            else if (value == "non-causal") { params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--rope-scaling"}, "{none,linear,yarn}",
        "RoPE frequency scaling method, defaults to linear unless specified by the model",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
            else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ));
    add_opt(llama_arg(
        {"--rope-scale"}, "N",
        "RoPE context scaling factor, expands context by a factor of N",
        [](gpt_params & params, const std::string & value) {
            params.rope_freq_scale = 1.0f / std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--rope-freq-base"}, "N",
        "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
        [](gpt_params & params, const std::string & value) {
            params.rope_freq_base = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--rope-freq-scale"}, "N",
        "RoPE frequency scaling factor, expands context by a factor of 1/N",
        [](gpt_params & params, const std::string & value) {
            params.rope_freq_scale = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--yarn-orig-ctx"}, "N",
        format("YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx),
        [](gpt_params & params, int value) {
            params.yarn_orig_ctx = value;
        }
    ));
    add_opt(llama_arg(
        {"--yarn-ext-factor"}, "N",
        format("YaRN: extrapolation mix factor (default: %.1f, 0.0 = full interpolation)", (double)params.yarn_ext_factor),
        [](gpt_params & params, const std::string & value) {
            params.yarn_ext_factor = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--yarn-attn-factor"}, "N",
        format("YaRN: scale sqrt(t) or attention magnitude (default: %.1f)", (double)params.yarn_attn_factor),
        [](gpt_params & params, const std::string & value) {
            params.yarn_attn_factor = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--yarn-beta-slow"}, "N",
        format("YaRN: high correction dim or alpha (default: %.1f)", (double)params.yarn_beta_slow),
        [](gpt_params & params, const std::string & value) {
            params.yarn_beta_slow = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"--yarn-beta-fast"}, "N",
        format("YaRN: low correction dim or beta (default: %.1f)", (double)params.yarn_beta_fast),
        [](gpt_params & params, const std::string & value) {
            params.yarn_beta_fast = std::stof(value);
        }
    ));
    add_opt(llama_arg(
        {"-gan", "--grp-attn-n"}, "N",
        format("group-attention factor (default: %d)", params.grp_attn_n),
        [](gpt_params & params, int value) {
            params.grp_attn_n = value;
        }
    ));
    add_opt(llama_arg(
        {"-gaw", "--grp-attn-w"}, "N",
        format("group-attention width (default: %.1f)", (double)params.grp_attn_w),
        [](gpt_params & params, int value) {
            params.grp_attn_w = value;
        }
    ));
    add_opt(llama_arg(
        {"-dkvc", "--dump-kv-cache"},
        "verbose print of the KV cache",
        [](gpt_params & params) {
            params.dump_kv_cache = true;
        }
    ));
    add_opt(llama_arg(
        {"-nkvo", "--no-kv-offload"},
        "disable KV offload",
        [](gpt_params & params) {
            params.no_kv_offload = true;
        }
    ));
    add_opt(llama_arg(
        {"-ctk", "--cache-type-k"}, "TYPE",
        format("KV cache data type for K (default: %s)", params.cache_type_k.c_str()),
        [](gpt_params & params, const std::string & value) {
            // TODO: get the type right here
            params.cache_type_k = value;
        }
    ));
    add_opt(llama_arg(
        {"-ctv", "--cache-type-v"}, "TYPE",
        format("KV cache data type for V (default: %s)", params.cache_type_v.c_str()),
        [](gpt_params & params, const std::string & value) {
            // TODO: get the type right here
            params.cache_type_v = value;
        }
    ));
    add_opt(llama_arg(
        {"--perplexity", "--all-logits"},
        format("return logits for all tokens in the batch (default: %s)", params.logits_all ? "true" : "false"),
        [](gpt_params & params) {
            params.logits_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--hellaswag"},
        "compute HellaSwag score over random tasks from datafile supplied with -f",
        [](gpt_params & params) {
            params.hellaswag = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--hellaswag-tasks"}, "N",
        format("number of tasks to use when computing the HellaSwag score (default: %zu)", params.hellaswag_tasks),
        [](gpt_params & params, int value) {
            params.hellaswag_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--winogrande"},
        "compute Winogrande score over random tasks from datafile supplied with -f",
        [](gpt_params & params) {
            params.winogrande = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--winogrande-tasks"}, "N",
        format("number of tasks to use when computing the Winogrande score (default: %zu)", params.winogrande_tasks),
        [](gpt_params & params, int value) {
            params.winogrande_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--multiple-choice"},
        "compute multiple choice score over random tasks from datafile supplied with -f",
        [](gpt_params & params) {
            params.multiple_choice = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--multiple-choice-tasks"}, "N",
        format("number of tasks to use when computing the multiple choice score (default: %zu)", params.multiple_choice_tasks),
        [](gpt_params & params, int value) {
            params.multiple_choice_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--kl-divergence"},
        "computes KL-divergence to logits provided via --kl-divergence-base",
        [](gpt_params & params) {
            params.kl_divergence = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--save-all-logits", "--kl-divergence-base"}, "FNAME",
        "set logits file",
        [](gpt_params & params, const std::string & value) {
            params.logits_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--ppl-stride"}, "N",
        format("stride for perplexity calculation (default: %d)", params.ppl_stride),
        [](gpt_params & params, int value) {
            params.ppl_stride = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"--ppl-output-type"}, "<0|1>",
        format("output type for perplexity calculation (default: %d)", params.ppl_output_type),
        [](gpt_params & params, int value) {
            params.ppl_output_type = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(llama_arg(
        {"-dt", "--defrag-thold"}, "N",
        format("KV cache defragmentation threshold (default: %.1f, < 0 - disabled)", (double)params.defrag_thold),
        [](gpt_params & params, const std::string & value) {
            params.defrag_thold = std::stof(value);
        }
    ).set_env("LLAMA_ARG_DEFRAG_THOLD"));
    add_opt(llama_arg(
        {"-np", "--parallel"}, "N",
        format("number of parallel sequences to decode (default: %d)", params.n_parallel),
        [](gpt_params & params, int value) {
            params.n_parallel = value;
        }
    ).set_env("LLAMA_ARG_N_PARALLEL"));
    add_opt(llama_arg(
        {"-ns", "--sequences"}, "N",
        format("number of sequences to decode (default: %d)", params.n_sequences),
        [](gpt_params & params, int value) {
            params.n_sequences = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PARALLEL}));
    add_opt(llama_arg(
        {"-cb", "--cont-batching"},
        format("enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.cont_batching = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONT_BATCHING"));
    add_opt(llama_arg(
        {"-nocb", "--no-cont-batching"},
        "disable continuous batching",
        [](gpt_params & params) {
            params.cont_batching = false;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_NO_CONT_BATCHING"));
    add_opt(llama_arg(
        {"--mmproj"}, "FILE",
        "path to a multimodal projector file for LLaVA. see examples/llava/README.md",
        [](gpt_params & params, const std::string & value) {
            params.mmproj = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LLAVA}));
    add_opt(llama_arg(
        {"--image"}, "FILE",
        "path to an image file. use with multimodal models. Specify multiple times for batching",
        [](gpt_params & params, const std::string & value) {
            params.image.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_LLAVA}));
#ifdef GGML_USE_RPC
    add_opt(llama_arg(
        {"--rpc"}, "SERVERS",
        "comma separated list of RPC servers",
        [](gpt_params & params, const std::string & value) {
            params.rpc_servers = value;
        }
    ));
#endif
    add_opt(llama_arg(
        {"--mlock"},
        "force system to keep model in RAM rather than swapping or compressing",
        [](gpt_params & params) {
            params.use_mlock = true;
        }
    ));
    add_opt(llama_arg(
        {"--no-mmap"},
        "do not memory-map model (slower load but may reduce pageouts if not using mlock)",
        [](gpt_params & params) {
            params.use_mmap = false;
        }
    ));
    add_opt(llama_arg(
        {"--numa"}, "TYPE",
        "attempt optimizations that help on some NUMA systems\n"
        "- distribute: spread execution evenly over all nodes\n"
        "- isolate: only spawn threads on CPUs on the node that execution started on\n"
        "- numactl: use the CPU map provided by numactl\n"
        "if run without this previously, it is recommended to drop the system page cache before using this\n"
        "see https://github.com/ggerganov/llama.cpp/issues/1437",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
            else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ));
    add_opt(llama_arg(
        {"-ngl", "--gpu-layers", "--n-gpu-layers"}, "N",
        "number of layers to store in VRAM",
        [](gpt_params & params, int value) {
            params.n_gpu_layers = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: not compiled with GPU offload support, --gpu-layers option will be ignored\n");
                fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
            }
        }
    ).set_env("LLAMA_ARG_N_GPU_LAYERS"));
    add_opt(llama_arg(
        {"-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
        "number of layers to store in VRAM for the draft model",
        [](gpt_params & params, int value) {
            params.n_gpu_layers_draft = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: not compiled with GPU offload support, --gpu-layers-draft option will be ignored\n");
                fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-sm", "--split-mode"}, "{none,layer,row}",
        "how to split the model across multiple GPUs, one of:\n"
        "- none: use one GPU only\n"
        "- layer (default): split layers and KV across GPUs\n"
        "- row: split rows across GPUs",
        [](gpt_params & params, const std::string & value) {
            std::string arg_next = value;
            if (arg_next == "none") {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            } else if (arg_next == "layer") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            } else if (arg_next == "row") {
#ifdef GGML_USE_SYCL
                fprintf(stderr, "warning: The split mode value:[row] is not supported by llama.cpp with SYCL. It's developing.\nExit!\n");
                exit(1);
#endif // GGML_USE_SYCL
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else {
                throw std::invalid_argument("invalid value");
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\n");
            }
        }
    ));
    add_opt(llama_arg(
        {"-ts", "--tensor-split"}, "N0,N1,N2,...",
        "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
        [](gpt_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    format("got %d input configs, but system only has %d devices", (int)split_arg.size(), (int)llama_max_devices())
                );
            }
            for (size_t i = 0; i < llama_max_devices(); ++i) {
                if (i < split_arg.size()) {
                    params.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params.tensor_split[i] = 0.0f;
                }
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\n");
            }
        }
    ));
    add_opt(llama_arg(
        {"-mg", "--main-gpu"}, "INDEX",
        format("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu),
        [](gpt_params & params, int value) {
            params.main_gpu = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\n");
            }
        }
    ));
    add_opt(llama_arg(
        {"--check-tensors"},
        format("check model tensor data for invalid values (default: %s)", params.check_tensors ? "true" : "false"),
        [](gpt_params & params) {
            params.check_tensors = true;
        }
    ));
    add_opt(llama_arg(
        {"--override-kv"}, "KEY=TYPE:VALUE",
        "advanced option to override model metadata by key. may be specified multiple times.\n"
        "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false",
        [](gpt_params & params, const std::string & value) {
            if (!string_parse_kv_override(value.c_str(), params.kv_overrides)) {
                throw std::runtime_error(format("error: Invalid type for KV override: %s\n", value.c_str()));
            }
        }
    ));
    add_opt(llama_arg(
        {"--lora"}, "FNAME",
        "path to LoRA adapter (can be repeated to use multiple adapters)",
        [](gpt_params & params, const std::string & value) {
            params.lora_adapters.push_back({ std::string(value), 1.0 });
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(llama_arg(
        {"--lora-scaled"}, "FNAME", "SCALE",
        "path to LoRA adapter with user defined scaling (can be repeated to use multiple adapters)",
        [](gpt_params & params, const std::string & fname, const std::string & scale) {
            params.lora_adapters.push_back({ fname, std::stof(scale) });
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(llama_arg(
        {"--control-vector"}, "FNAME",
        "add a control vector\nnote: this argument can be repeated to add multiple control vectors",
        [](gpt_params & params, const std::string & value) {
            params.control_vectors.push_back({ 1.0f, value, });
        }
    ));
    add_opt(llama_arg(
        {"--control-vector-scaled"}, "FNAME", "SCALE",
        "add a control vector with user defined scaling SCALE\n"
        "note: this argument can be repeated to add multiple scaled control vectors",
        [](gpt_params & params, const std::string & fname, const std::string & scale) {
            params.control_vectors.push_back({ std::stof(scale), fname });
        }
    ));
    add_opt(llama_arg(
        {"--control-vector-layer-range"}, "START", "END",
        "layer range to apply the control vector(s) to, start and end inclusive",
        [](gpt_params & params, const std::string & start, const std::string & end) {
            params.control_vector_layer_start = std::stoi(start);
            params.control_vector_layer_end = std::stoi(end);
        }
    ));
    add_opt(llama_arg(
        {"-a", "--alias"}, "STRING",
        "set alias for model name (to be used by REST API)",
        [](gpt_params & params, const std::string & value) {
            params.model_alias = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"-m", "--model"}, "FNAME",
        ex == LLAMA_EXAMPLE_EXPORT_LORA
            ? std::string("model path from which to load base model")
            : format(
                "model path (default: `models/$filename` with filename from `--hf-file` "
                "or `--model-url` if set, otherwise %s)", DEFAULT_MODEL_PATH
            ),
        [](gpt_params & params, const std::string & value) {
            params.model = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}).set_env("LLAMA_ARG_MODEL"));
    add_opt(llama_arg(
        {"-md", "--model-draft"}, "FNAME",
        "draft model for speculative decoding (default: unused)",
        [](gpt_params & params, const std::string & value) {
            params.model_draft = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(llama_arg(
        {"-mu", "--model-url"}, "MODEL_URL",
        "model download url (default: unused)",
        [](gpt_params & params, const std::string & value) {
            params.model_url = value;
        }
    ).set_env("LLAMA_ARG_MODEL_URL"));
    add_opt(llama_arg(
        {"-hfr", "--hf-repo"}, "REPO",
        "Hugging Face model repository (default: unused)",
        [](gpt_params & params, const std::string & value) {
            params.hf_repo = value;
        }
    ).set_env("LLAMA_ARG_HF_REPO"));
    add_opt(llama_arg(
        {"-hff", "--hf-file"}, "FILE",
        "Hugging Face model file (default: unused)",
        [](gpt_params & params, const std::string & value) {
            params.hf_file = value;
        }
    ).set_env("LLAMA_ARG_HF_FILE"));
    add_opt(llama_arg(
        {"-hft", "--hf-token"}, "TOKEN",
        "Hugging Face access token (default: value from HF_TOKEN environment variable)",
        [](gpt_params & params, const std::string & value) {
            params.hf_token = value;
        }
    ).set_env("HF_TOKEN"));
    add_opt(llama_arg(
        {"--context-file"}, "FNAME",
        "file to load context from (repeat to specify multiple files)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            params.context_files.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(llama_arg(
        {"--chunk-size"}, "N",
        format("minimum length of embedded text chunks (default: %d)", params.chunk_size),
        [](gpt_params & params, int value) {
            params.chunk_size = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(llama_arg(
        {"--chunk-separator"}, "STRING",
        format("separator between chunks (default: '%s')", params.chunk_separator.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.chunk_separator = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(llama_arg(
        {"--junk"}, "N",
        format("number of times to repeat the junk text (default: %d)", params.n_junk),
        [](gpt_params & params, int value) {
            params.n_junk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(llama_arg(
        {"--pos"}, "N",
        format("position of the passkey in the junk text (default: %d)", params.i_pos),
        [](gpt_params & params, int value) {
            params.i_pos = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(llama_arg(
        {"-o", "--output", "--output-file"}, "FNAME",
        format("output file (default: '%s')",
            ex == LLAMA_EXAMPLE_EXPORT_LORA
                ? params.lora_outfile.c_str()
                : ex == LLAMA_EXAMPLE_CVECTOR_GENERATOR
                    ? params.cvector_outfile.c_str()
                    : params.out_file.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.out_file = value;
            params.cvector_outfile = value;
            params.lora_outfile = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_CVECTOR_GENERATOR, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(llama_arg(
        {"-ofreq", "--output-frequency"}, "N",
        format("output the imatrix every N iterations (default: %d)", params.n_out_freq),
        [](gpt_params & params, int value) {
            params.n_out_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"--save-frequency"}, "N",
        format("save an imatrix copy every N iterations (default: %d)", params.n_save_freq),
        [](gpt_params & params, int value) {
            params.n_save_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"--process-output"},
        format("collect data for the output tensor (default: %s)", params.process_output ? "true" : "false"),
        [](gpt_params & params) {
            params.process_output = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"--no-ppl"},
        format("do not compute perplexity (default: %s)", params.compute_ppl ? "true" : "false"),
        [](gpt_params & params) {
            params.compute_ppl = false;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"--chunk", "--from-chunk"}, "N",
        format("start processing the input from chunk N (default: %d)", params.i_chunk),
        [](gpt_params & params, int value) {
            params.i_chunk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(llama_arg(
        {"-pps"},
        format("is the prompt shared across parallel sequences (default: %s)", params.is_pp_shared ? "true" : "false"),
        [](gpt_params & params) {
            params.is_pp_shared = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(llama_arg(
        {"-npp"}, "n0,n1,...",
        "number of prompt tokens",
        [](gpt_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pp.insert(params.n_pp.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(llama_arg(
        {"-ntg"}, "n0,n1,...",
        "number of text generation tokens",
        [](gpt_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_tg.insert(params.n_tg.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(llama_arg(
        {"-npl"}, "n0,n1,...",
        "number of parallel prompts",
        [](gpt_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pl.insert(params.n_pl.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(llama_arg(
        {"--embd-normalize"}, "N",
        format("normalisation for embendings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", params.embd_normalize),
        [](gpt_params & params, int value) {
            params.embd_normalize = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--embd-output-format"}, "FORMAT",
        "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix",
        [](gpt_params & params, const std::string & value) {
            params.embd_out = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--embd-separator"}, "STRING",
        "separator of embendings (default \\n) for example \"<#sep#>\"",
        [](gpt_params & params, const std::string & value) {
            params.embd_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(llama_arg(
        {"--host"}, "HOST",
        format("ip address to listen (default: %s)", params.hostname.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.hostname = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_HOST"));
    add_opt(llama_arg(
        {"--port"}, "PORT",
        format("port to listen (default: %d)", params.port),
        [](gpt_params & params, int value) {
            params.port = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PORT"));
    add_opt(llama_arg(
        {"--path"}, "PATH",
        format("path to serve static files from (default: %s)", params.public_path.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.public_path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--embedding", "--embeddings"},
        format("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", params.embedding ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_EMBEDDINGS"));
    add_opt(llama_arg(
        {"--api-key"}, "KEY",
        "API key to use for authentication (default: none)",
        [](gpt_params & params, const std::string & value) {
            params.api_keys.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_API_KEY"));
    add_opt(llama_arg(
        {"--api-key-file"}, "FNAME",
        "path to file containing API keys (default: none)",
        [](gpt_params & params, const std::string & value) {
            std::ifstream key_file(value);
            if (!key_file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string key;
            while (std::getline(key_file, key)) {
                if (!key.empty()) {
                        params.api_keys.push_back(key);
                }
            }
            key_file.close();
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--ssl-key-file"}, "FNAME",
        "path to file a PEM-encoded SSL private key",
        [](gpt_params & params, const std::string & value) {
            params.ssl_file_key = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--ssl-cert-file"}, "FNAME",
        "path to file a PEM-encoded SSL certificate",
        [](gpt_params & params, const std::string & value) {
            params.ssl_file_cert = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"-to", "--timeout"}, "N",
        format("server read/write timeout in seconds (default: %d)", params.timeout_read),
        [](gpt_params & params, int value) {
            params.timeout_read  = value;
            params.timeout_write = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--threads-http"}, "N",
        format("number of threads used to process HTTP requests (default: %d)", params.n_threads_http),
        [](gpt_params & params, int value) {
            params.n_threads_http = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_THREADS_HTTP"));
    add_opt(llama_arg(
        {"-spf", "--system-prompt-file"}, "FNAME",
        "set a file to load a system prompt (initial prompt of all slots), this is useful for chat applications",
        [](gpt_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string system_prompt;
            std::copy(
                        std::istreambuf_iterator<char>(file),
                        std::istreambuf_iterator<char>(),
                        std::back_inserter(system_prompt)
                        );
            params.system_prompt = system_prompt;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--metrics"},
        format("enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.endpoint_metrics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_METRICS"));
    add_opt(llama_arg(
        {"--no-slots"},
        format("disables slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.endpoint_slots = false;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_NO_ENDPOINT_SLOTS"));
    add_opt(llama_arg(
        {"--slot-save-path"}, "PATH",
        "path to save slot kv cache (default: disabled)",
        [](gpt_params & params, const std::string & value) {
            params.slot_save_path = value;
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.slot_save_path.empty() && params.slot_save_path[params.slot_save_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.slot_save_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--chat-template"}, "JINJA_TEMPLATE",
        "set custom jinja chat template (default: template taken from model's metadata)\n"
        "if suffix/prefix are specified, template will be disabled\n"
        "only commonly used templates are accepted:\nhttps://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template",
        [](gpt_params & params, const std::string & value) {
            if (!llama_chat_verify_template(value)) {
                throw std::runtime_error(format(
                    "error: the supplied chat template is not supported: %s\n"
                    "note: llama.cpp does not use jinja parser, we only support commonly used templates\n",
                    value.c_str()
                ));
            }
            params.chat_template = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CHAT_TEMPLATE"));
    add_opt(llama_arg(
        {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
        format("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity),
        [](gpt_params & params, const std::string & value) {
            params.slot_prompt_similarity = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--lora-init-without-apply"},
        format("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora_init_without_apply ? "enabled" : "disabled"),
        [](gpt_params & params) {
            params.lora_init_without_apply = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(llama_arg(
        {"--simple-io"},
        "use basic IO for better compatibility in subprocesses and limited consoles",
        [](gpt_params & params) {
            params.simple_io = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
    add_opt(llama_arg(
        {"-ld", "--logdir"}, "LOGDIR",
        "path under which to save YAML logs (no logging if unset)",
        [](gpt_params & params, const std::string & value) {
            params.logdir = value;

            if (params.logdir.back() != DIRECTORY_SEPARATOR) {
                params.logdir += DIRECTORY_SEPARATOR;
            }
        }
    ));
    add_opt(llama_arg(
        {"--positive-file"}, "FNAME",
        format("positive prompts file, one prompt per line (default: '%s')", params.cvector_positive_file.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.cvector_positive_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--negative-file"}, "FNAME",
        format("negative prompts file, one prompt per line (default: '%s')", params.cvector_negative_file.c_str()),
        [](gpt_params & params, const std::string & value) {
            params.cvector_negative_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--pca-batch"}, "N",
        format("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", params.n_pca_batch),
        [](gpt_params & params, int value) {
            params.n_pca_batch = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--pca-iter"}, "N",
        format("number of iterations used for PCA (default: %d)", params.n_pca_iterations),
        [](gpt_params & params, int value) {
            params.n_pca_iterations = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--method"}, "{pca, mean}",
        "dimensionality reduction method to be used (default: pca)",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "pca") { params.cvector_dimre_method = DIMRE_METHOD_PCA; }
            else if (value == "mean") { params.cvector_dimre_method = DIMRE_METHOD_MEAN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(llama_arg(
        {"--output-format"}, "{md,jsonl}",
        "output format for batched-bench results (default: md)",
        [](gpt_params & params, const std::string & value) {
            /**/ if (value == "jsonl") { params.batched_bench_output_jsonl = true; }
            else if (value == "md") { params.batched_bench_output_jsonl = false; }
            else { std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(llama_arg(
        {"--log-disable"},
        "Log disable",
        [](gpt_params &) {
            gpt_log_pause(gpt_log_main());
        }
    ));
    add_opt(llama_arg(
        {"--log-file"}, "FNAME",
        "Log to file",
        [](gpt_params &, const std::string & value) {
            gpt_log_set_file(gpt_log_main(), value.c_str());
        }
    ));
    add_opt(llama_arg(
        {"--log-colors"},
        "Enable colored logging",
        [](gpt_params &) {
            gpt_log_set_colors(gpt_log_main(), true);
        }
    ).set_env("LLAMA_LOG_COLORS"));
    add_opt(llama_arg(
        {"-v", "--verbose", "--log-verbose"},
        "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
        [](gpt_params & params) {
            params.verbosity = INT_MAX;
            gpt_log_set_verbosity_thold(INT_MAX);
        }
    ));
    add_opt(llama_arg(
        {"-lv", "--verbosity", "--log-verbosity"}, "N",
        "Set the verbosity threshold. Messages with a higher verbosity will be ignored.",
        [](gpt_params & params, int value) {
            params.verbosity = value;
            gpt_log_set_verbosity_thold(value);
        }
    ).set_env("LLAMA_LOG_VERBOSITY"));
    add_opt(llama_arg(
        {"--log-prefix"},
        "Enable prefx in log messages",
        [](gpt_params &) {
            gpt_log_set_prefix(gpt_log_main(), true);
        }
    ).set_env("LLAMA_LOG_PREFIX"));
    add_opt(llama_arg(
        {"--log-timestamps"},
        "Enable timestamps in log messages",
        [](gpt_params &) {
            gpt_log_set_timestamps(gpt_log_main(), true);
        }
    ).set_env("LLAMA_LOG_TIMESTAMPS"));

    return ctx_arg;
}
