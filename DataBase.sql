-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Tempo de geração: 28-Out-2025 às 16:30
-- Versão do servidor: 10.4.32-MariaDB
-- versão do PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Banco de dados: `valores`
--

-- --------------------------------------------------------

--
-- Estrutura da tabela `ai_bpm`
--

CREATE TABLE `ai_bpm` (
  `paciente_id` int(10) UNSIGNED NOT NULL,
  `p_bradi` decimal(5,4) NOT NULL,
  `p_normal` decimal(5,4) NOT NULL,
  `p_taqui` decimal(5,4) NOT NULL,
  `label` enum('bradicardia','normal','taquicardia') NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Estrutura da tabela `ai_stress`
--

CREATE TABLE `ai_stress` (
  `paciente_id` int(10) UNSIGNED NOT NULL,
  `p_normal` decimal(5,4) NOT NULL,
  `p_stress` decimal(5,4) NOT NULL,
  `label` enum('normal','stress') NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Estrutura da tabela `config`
--

CREATE TABLE `config` (
  `cfg_key` varchar(64) NOT NULL,
  `cfg_value` int(255) NOT NULL,
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Estrutura da tabela `referencias`
--

CREATE TABLE `referencias` (
  `processo_id` int(10) NOT NULL,
  `data_hora` timestamp NOT NULL DEFAULT current_timestamp(),
  `medico_nome` varchar(100) NOT NULL,
  `medico_numero` int(50) NOT NULL,
  `paciente_nome` varchar(100) NOT NULL,
  `paciente_id` int(50) NOT NULL,
  `informacoes` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Estrutura da tabela `sensores`
--

CREATE TABLE `sensores` (
  `id` int(11) NOT NULL,
  `paciente_id` int(11) NOT NULL,
  `pulso` float NOT NULL,
  `respiracao` float NOT NULL,
  `gsr` float NOT NULL,
  `temperatura` float NOT NULL,
  `ax` float NOT NULL,
  `ay` float NOT NULL,
  `az` float NOT NULL,
  `data_hora` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Acionadores `sensores`
--
DELIMITER $$
CREATE TRIGGER `sensores_force_paciente_ins` BEFORE INSERT ON `sensores` FOR EACH ROW BEGIN
  DECLARE pid BIGINT;
  SELECT CAST(cfg_value AS UNSIGNED)
    INTO pid
    FROM config
    WHERE cfg_key='paciente_ativo'
    LIMIT 1;

  IF pid IS NOT NULL THEN
    SET NEW.paciente_id = pid;
  END IF;
END
$$
DELIMITER ;
DELIMITER $$
CREATE TRIGGER `sensores_force_paciente_upd` BEFORE UPDATE ON `sensores` FOR EACH ROW BEGIN
  DECLARE pid BIGINT;
  SELECT CAST(cfg_value AS UNSIGNED)
    INTO pid
    FROM config
    WHERE cfg_key='paciente_ativo'
    LIMIT 1;

  IF pid IS NOT NULL THEN
    SET NEW.paciente_id = pid;
  END IF;
END
$$
DELIMITER ;

--
-- Índices para tabelas despejadas
--

--
-- Índices para tabela `ai_stress`
--
ALTER TABLE `ai_stress`
  ADD KEY `ix_ai_stress_paciente_created` (`paciente_id`,`created_at`);

--
-- Índices para tabela `config`
--
ALTER TABLE `config`
  ADD PRIMARY KEY (`cfg_key`);

--
-- Índices para tabela `sensores`
--
ALTER TABLE `sensores`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT de tabelas despejadas
--

--
-- AUTO_INCREMENT de tabela `sensores`
--
ALTER TABLE `sensores`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
