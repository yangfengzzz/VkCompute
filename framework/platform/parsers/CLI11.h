//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <unordered_map>

#include <CLI/CLI.hpp>

#include "help_formatter.h"
#include "platform/parser.h"
#include "platform/plugins/plugin.h"

namespace vox {
struct CLI11CommandContextState

{
    std::string group_name = "";
};

class CLI11CommandContext : public CommandParserContext {
public:
    CLI11CommandContext(CLI::App *cli, const CLI11CommandContextState &state = {});

    virtual ~CLI11CommandContext() = default;

    bool has_group_name() const;
    const std::string &get_group_name() const;
    CLI11CommandContextState get_state() const;

    CLI::App *cli11;

private:
    CLI11CommandContextState _state;
};

class CLI11CommandParser : public CommandParser {
public:
    CLI11CommandParser(const std::string &name, const std::string &description, const std::vector<std::string> &args);
    virtual ~CLI11CommandParser() = default;

    virtual std::vector<std::string> help() const override;

    virtual bool parse(const std::vector<Plugin *> &plugins) override;

    virtual bool parse(const std::vector<Command *> &commands) override;

protected:
#define CAST(type) virtual void parse(CommandParserContext *context, type *command) override;
    CAST(CommandGroup);
    CAST(SubCommand);
    CAST(PositionalCommand);
    CAST(FlagCommand);
#undef CAST

    void parse(CLI11CommandContext *context, CommandGroup *command);
    void parse(CLI11CommandContext *context, SubCommand *command);
    void parse(CLI11CommandContext *context, PositionalCommand *command);
    void parse(CLI11CommandContext *context, FlagCommand *command);

    virtual bool contains(Command *command) const override;

private:
    std::vector<const char *> _args;

    std::unique_ptr<CLI::App> _cli11;
    std::unordered_map<Command *, CLI::Option *> _options;
    std::unordered_map<Command *, CLI::App *> _sub_commands;
    std::unordered_map<Plugin *, std::shared_ptr<CLI::App>> _option_groups;
    std::shared_ptr<HelpFormatter> _formatter;

    virtual std::vector<std::string> get_command_value(Command *command) const override;

    bool cli11_parse(CLI::App *app);
};
}// namespace vox